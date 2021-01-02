
import tensorflow as tf
from basic_dqn_FP.utils import init_cnn, init_tdcnn_rnn
def init_network(config):
    if config.network == 'cnn':
        network = init_cnn(config)
    if config.network == 'tdcnn_rnn':
        network = init_tdcnn_rnn(config)

    return network

class Network(tf.Module):
    def __init__(self, config):
        self.config = config
        self.agent_ids = [a for a in range(config.num_agents)]
        # self.env = env
        self.model = init_network(config)
        self.target_model = init_network(config)
        self.model.summary()
        tf.keras.utils.plot_model(self.model, to_file='./model.png')

        if self.config.dueling:
            self.agent_heads = self.build_agent_heads_dueling()
            self.target_agent_heads = self.build_agent_heads_dueling()
        else:
            self.agent_heads = self.build_agent_heads()
            self.target_agent_heads = self.build_agent_heads()

        self.agent_heads[0].summary()
        tf.keras.utils.plot_model(self.agent_heads[0], to_file='./agent_heads_model.png')

        if config.load_path is not None:
            self.load_models(config.load_path)

        self.one_hot_agents = tf.expand_dims(tf.one_hot(self.agent_ids, len(self.agent_ids), dtype=tf.float32), axis=1)
        print(f'self.onehot_agent.shape is {self.one_hot_agents.shape}')

        self.initialize_dummy_variabels()

    def build_agent_heads(self):
        """

        :return: list of heads for agents

            - gets tensorflow model and adds heads for each agent
        """
        input_shape = self.model.output_shape[-1]
        heads = []
        inputs = tf.keras.layers.Input(input_shape)
        for a in self.agent_ids:
            name = 'head_agent_' + str(a)
            head_a = tf.keras.layers.Dense(units=self.config.num_actions, activation=None,
                                           kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                           bias_initializer=tf.keras.initializers.Constant(0.0),
                                           name=name)(inputs)
            head_a = tf.keras.Model(inputs=inputs, outputs=head_a)
            heads.append(head_a)

        return heads

    def build_agent_heads_dueling(self):
        """

        :return: list of heads for agents

            - gets tensorflow model and adds heads for each agent
        """
        input_shape = self.model.output_shape[-1]
        print(input_shape)
        heads = []
        inputs = tf.keras.layers.Input(input_shape)
        for a in self.agent_ids:
            name = 'head_agent_' + str(a)
            with tf.name_scope(f'action_value_{name}'):
                action_head_a = tf.keras.layers.Dense(units=self.config.num_actions, activation=None,
                                                      kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                                      bias_initializer=tf.keras.initializers.Constant(0.0),
                                                      name='action_' + name)(inputs)

            with tf.name_scope(f'state_value_{name}'):
                state_head_a = tf.keras.layers.Dense(units=1, activation=None,
                                                     kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                                     bias_initializer=tf.keras.initializers.Constant(0.0),
                                                     name='state_' + name)(inputs)

            action_scores_mean = tf.reduce_mean(action_head_a, 1)
            action_scores_centered = action_head_a - tf.expand_dims(action_scores_mean, 1)
            head_a = state_head_a + action_scores_centered

            head_a = tf.keras.Model(inputs=inputs, outputs=head_a)
            heads.append(head_a)

        return heads

    @tf.function
    def value(self, obs, fps, agent_id):
        inputs_a = {0: obs,
                    1: fps}

        fc_values = self.model(inputs_a)
        q_t = self.agent_heads[agent_id](fc_values)

        return q_t

    @tf.function
    def step(self, obs):
        """

        :param obs: list observations one for each agent
        :return: actions: list of actions and fingerperintschosen by agents based on observation one for each agent
        """

        actions = []
        fps = []
        for a in self.agent_ids:
            if self.config.network == 'tdcnn_rnn':
                inputs = {0: tf.expand_dims(tf.expand_dims(obs[a], 0), 0),
                          1: tf.expand_dims(self.dummy_fps, 0)}
            else:
                inputs = {0: tf.expand_dims(obs[a], 0),
                          1: self.dummy_fps}

            fc_values = self.model(inputs)
            q_values = self.agent_heads[a](fc_values)
            fps.append(q_values.numpy()[0])
            deterministic_actions = tf.argmax(q_values, axis=1)

            actions.append(deterministic_actions.numpy()[0])
        return actions, fps

    @tf.function
    def last_value(self, obs):
        """

        :param obs: list observations one for each agent
        :return: best values based on Q-Learning formula max Q(s',a')
        """

        values = []
        for a in self.agent_ids:
            if self.config.network == 'tdcnn_rnn':
                inputs = {0: tf.expand_dims(tf.expand_dims(obs[a], 0), 0),
                          1: tf.expand_dims(self.dummy_fps, 0)}
            else:
                inputs = {0: tf.expand_dims(obs[a], 0),
                          1: self.dummy_fps}

            fc_values = self.target_model(inputs)
            q_values = self.target_agent_heads[a](fc_values)

            if self.config.double_q:
                fc_values_using_online_net = self.model(inputs)
                q_values_using_online_net = self.agent_heads[a](fc_values_using_online_net)
                q_value_best_using_online_net = tf.argmax(q_values_using_online_net, 1)
                q_tp1_best = tf.reduce_sum(
                    q_values * tf.one_hot(q_value_best_using_online_net, self.config.num_actions, dtype=tf.float32), 1)
            else:
                q_tp1_best = tf.reduce_max(q_values, 1)

            values.append(q_tp1_best.numpy()[0])

        return values

    def initialize_dummy_variabels(self):
        self.dummy_fps = tf.zeros((1, self.config.fp_shape))
        self.dummy_done_mask = tf.zeros((1, 1))


    @tf.function(autograph=False)
    def update_target(self):
        for var, var_target in zip(self.model.trainable_variables, self.target_model.trainable_variables):
            var_target.assign(var)

        vars_, target_vars = [], []
        for a in self.agent_ids:
            vars_.extend(self.agent_heads[a].trainable_variables)
            target_vars.extend(self.target_agent_heads[a].trainable_variables)

        for var, var_target in zip(vars_, target_vars):
            var_target.assign(var)

    @tf.function(autograph=False)
    def soft_update_target(self):
        for var, var_target in zip(self.model.trainable_variables, self.target_model.trainable_variables):
            var_target.assign(self.config.tau * var + (1.0 - self.config.tau) * var_target)

        vars, target_vars = [], []
        for a in self.agent_ids:
            vars.extend(self.agent_heads[a].trainable_variables)
            target_vars.extend(self.target_agent_heads[a].trainable_variables)

        for var, var_target in zip(vars, target_vars):
            var_target.assign(self.config.tau * var + (1.0 - self.config.tau) * var_target)

    def save(self, save_path):
        self.model.save_weights(f'{save_path}/value_network.h5')
        for a in self.agent_ids:
            self.agent_heads[a].save_weights(f'{save_path}/agent_{a}_head.h5')

    def load(self, load_path):
        self.model.load_weights(f'{load_path}/value_network.h5')
        self.target_model.load_weights(f'{load_path}/value_network.h5')
        for a in self.agent_ids:
            self.agent_heads[a].load_weights(f'{load_path}/agent_{a}_head.h5')
            self.target_agent_heads[a].load_weights(f'{load_path}/agent_{a}_head.h5')



