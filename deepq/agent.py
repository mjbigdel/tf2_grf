import numpy as np
import tensorflow as tf
from deepq.utils import huber_loss


# Agent
class Agent:
    def __init__(self, config, model, target_model, agent_id):
        super().__init__()
        self.agent_id = agent_id
        self.config = config
        self.model = model
        self.model.summary()
        self.target_model = target_model

    def create_fingerprint(self, fps, t):
        # TODO
        fps = []
        if self.config.num_agents > 1:
            fp = fps[:self.agent_id]
            fp.extend(fps[self.agent_id + 1:])
            fp_a = np.concatenate((fp, [[self.exploration.value(t) * 100, t]]), axis=None)
            fps.append(fp_a)
        return fps

    @tf.function
    def max_value(self, obs):
        """
        :param obs: list observations one for each agent
        :return: best values based on Q-Learning formula maxQ(s',a')
        """
        obs = tf.expand_dims(obs, axis=0)
        q_tp1 = self.target_model(obs)
        # print(f' q_tp1 {q_tp1}')

        if self.config.double_q:
            q_values_using_online_net = self.model(obs)
            q_value_best_using_online_net = tf.argmax(q_values_using_online_net, 1)
            q_tp1_best = tf.reduce_sum(
                q_tp1 * tf.one_hot(q_value_best_using_online_net, self.config.num_actions, dtype=tf.float32), 1)
        else:
            q_tp1_best = tf.reduce_max(q_tp1, 1)

        return q_tp1_best.numpy()[0]

    @tf.function
    def greedy_action(self, obs):
        """
        :param obs: observation for agent_id
        :return: action, q_values as fp for agent_id
        """
        # print(f' greedy_action obs.shape {obs.shape}')
        obs = tf.expand_dims(obs, axis=0)
        q_values = self.model(obs)
        deterministic_actions = tf.argmax(q_values, axis=1)

        return deterministic_actions.numpy()[0], q_values.numpy()[0]

    @tf.function()
    def compute_loss(self, obses_t, actions, rewards, dones, weights, fps=None):
        # print(f' obs.shape {obses_t.shape}')
        q_t = self.model(obses_t)

        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(actions, self.config.num_actions, dtype=tf.float32), 1)
        # print(f'q_t_selected.shape is {q_t_selected.shape}')

        td_error = q_t_selected - tf.stop_gradient(rewards)

        errors = huber_loss(td_error)
        weighted_loss = tf.reduce_mean(weights * errors)

        return weighted_loss, td_error

    @tf.function(autograph=False)
    def update_target(self):
        for var, var_target in zip(self.model.trainable_variables, self.target_model.trainable_variables):
            var_target.assign(var)

    @tf.function(autograph=False)
    def soft_update_target(self):
        for var, var_target in zip(self.model.trainable_variables, self.target_model.trainable_variables):
            var_target.assign(self.config.tau * var + (1.0 - self.config.tau) * var_target)

    def save(self):
        self.model.save_weights(f'{self.config.save_path}/model_agent_{self.agent_id}.h5')

    def load(self):
        self.model.load_weights(f'{self.config.load_path}/model_agent_{self.agent_id}.h5')
        self.target_model.load_weights(f'{self.config.load_path}/model_agent_{self.agent_id}.h5')
