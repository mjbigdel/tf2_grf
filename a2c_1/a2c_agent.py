import tensorflow as tf
from common.policies import PolicyWithValue
from a2c_1.utils import InverseLinearTimeDecay

class Agent(tf.keras.Model):

    """
    We use this class to :
        __init__:
        - Creates the step_model
        - Creates the train_model

        train():
        - Make the training part (feedforward and retropropagation of gradients)

        save/load():
        - Save load the model
    """
    def __init__(self, *, ac_space, policy_network, nupdates,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6)):

        super(Agent, self).__init__(name='A2CModel')
        self.train_model = PolicyWithValue(ac_space, policy_network, value_network=None, estimate_q=False)
        # lr_schedule = InverseLinearTimeDecay(initial_learning_rate=lr, nupdates=nupdates)
        # self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, rho=alpha, epsilon=epsilon)
        self.optimizer = tf.keras.optimizers.Adam(lr)
        # self.train_model.value_model.compile(self.optimizer)
        # self.train_model.value_model.summary()
        # tf.keras.utils.plot_model(self.train_model.value_model, to_file='./model.png')

        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.step = self.train_model.step
        self.value = self.train_model.value
        self.initial_state = self.train_model.initial_state

    @tf.function
    def train(self, obs, states, rewards, masks, actions, values):
        advs = rewards - values
        with tf.GradientTape() as tape:
            policy_latent = self.train_model.policy_network(obs)
            pd, _ = self.train_model.pdtype.pdfromlatent(policy_latent)
            neglogpac = pd.neglogp(actions)
            entropy = tf.reduce_mean(pd.entropy())
            vpred = self.train_model.value(obs)
            vf_loss = tf.reduce_mean(tf.square(vpred - rewards))
            pg_loss = tf.reduce_mean(advs * neglogpac)
            loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

        # var_list = tape.watched_variables()
        var_list = self.train_model.policy_network.trainable_variables
        grads = tape.gradient(loss, var_list)
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        grads_and_vars = list(zip(grads, var_list))
        self.optimizer.apply_gradients(grads_and_vars)

        return pg_loss, vf_loss, entropy
    

