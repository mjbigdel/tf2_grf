import numpy as np
import tensorflow as tf
from distdeepq.utils import huber_loss


# Agent
class Agent:
    def __init__(self, config, model, target_model, agent_id):
        # super().__init__()
        self.agent_id = agent_id
        self.config = config
        self.model = model
        self.model.summary()
        self.target_model = target_model
        self.loss_dist = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        self.v_min = -5.0
        self.v_max = 5.0
        self.delta_z = float(self.v_max - self.v_min) / (self.config.atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.config.atoms)]

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
            q_value_best_using_online_net = tf.argmax(q_values_using_online_net, 1)  # shape=(1,)
            q_tp1_best = tf.reduce_sum(
                q_tp1 * tf.one_hot(q_value_best_using_online_net, self.config.num_actions, dtype=tf.float32), 1)
        else:
            q_tp1_best = tf.reduce_max(q_tp1, 1)

        return q_tp1_best.numpy()[0]

    @tf.function
    def max_value_dist(self, obs):
        """
        :param obs: list observations one for each agent
        :return: best values based on Q-Learning formula maxQ(s',a')
        """
        obs = tf.expand_dims(obs, axis=0)
        # print(f' obs.shape {obs.shape}')
        zz = self.target_model(obs)
        # print(f' zz.shape {zz.shape}')
        # z_concat = np.vstack(zz)
        # print(f' z_concat.shape {z_concat.shape}')
        q_tp1 = tf.reduce_sum(tf.math.multiply(zz, self.z), axis=-1)
        # q_tp1 = self.target_model(obs)
        # print(f' q_tp1.shape {q_tp1.shape}')

        if self.config.double_q:
            z_using_online_net = self.model(obs)
            # print(f' z_using_online_net.shape {z_using_online_net.shape}')
            # z_using_online_net_concat = np.vstack(z_using_online_net)
            # print(f' z_using_online_net_concat.shape {z_using_online_net_concat.shape}')
            q_values_using_online_net = tf.reduce_sum(tf.math.multiply(z_using_online_net, self.z), axis=-1)
            # print(f' q_values_using_online_net.shape {q_values_using_online_net.shape}')
            q_value_best_using_online_net = tf.argmax(q_values_using_online_net, 1)
            # print(f' q_value_best_using_online_net.shape {q_value_best_using_online_net.shape}')
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

    @tf.function
    def greedy_action_dist(self, obs):
        """
        :param obs: observation for agent_id
        :return: action, q_values as fp for agent_id
        """
        # print(f' greedy_action obs.shape {obs.shape}')
        obs = tf.expand_dims(obs, axis=0)
        zz = self.model(obs)  # shape (1, 19, 8)
        # print(f' zz.shape {zz.shape}')
        # z_concat = np.vstack(zz)
        # print(f' z_concat.shape {z_concat.shape}')
        q_values = tf.reduce_sum(tf.math.multiply(zz, self.z), axis=-1)
        # print(f' q_values.shape {q_values.shape}')
        deterministic_actions = tf.argmax(q_values, 1)

        # print(f' deterministic_actions.numpy() {deterministic_actions.numpy()}')
        # print(f' q_values.numpy() {q_values.numpy()}')
        return deterministic_actions.numpy()[0], q_values.numpy()[0]

    @tf.function()
    def compute_loss(self, obses_t, actions, rewards, obs_tp1, dones, weights, fps=None):
        # print(f' obs.shape {obses_t.shape}')
        q_t = self.model(obses_t)

        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(actions, self.config.num_actions, dtype=tf.float32), 1)
        # print(f'q_t_selected.shape is {q_t_selected.shape}')

        td_error = q_t_selected - tf.stop_gradient(rewards)

        errors = huber_loss(td_error)
        weighted_loss = tf.reduce_mean(weights * errors)

        return weighted_loss, td_error

    @tf.function()
    def compute_loss_dist(self, obses_t, actions, rewards, obs_tp1, dones, weights, fps=None):
        # print(f' obses_t.shape {obses_t.shape}')
        # rewards = tf.stop_gradient(rewards)

        logits = self.model(obses_t)
        # print(f' logits.shape {logits.shape}')

        zz = self.model(obs_tp1)
        # print(f' zz.shape {zz.shape}')
        q = tf.reduce_sum(tf.math.multiply(zz, self.z), axis=-1)
        # print(f' q.shape {q.shape}')
        next_actions = tf.argmax(q, axis=1)  # a* in C51 algo

        zz_ = self.target_model(obs_tp1)
        # print(f' zz_.shape {zz_.shape}')
        # z_concat = tf.stack(z)

        m_prob = [np.zeros((self.config.num_actions, self.config.atoms), dtype=np.float32)
                  for _ in range(self.config.batch_size)]

        for i in range(self.config.batch_size):
            if dones[i]:
                Tz = min(self.v_max, max(self.v_min, rewards[i]))
                bj = (Tz - self.v_min) / self.delta_z
                l, u = tf.math.floor(bj), tf.math.ceil(bj)

                m_prob[i][actions[i]][int(l)] += (u - bj)
                m_prob[i][actions[i]][int(u)] += (bj - l)
            else:
                for j in range(self.config.atoms):
                    # compute the projection of Tzj onto the support {zi}
                    Tz = min(self.v_max, max(self.v_min, rewards[i] + self.config.gamma * self.z[j]))
                    bj = (Tz - self.v_min) / self.delta_z
                    l, u = tf.math.floor(bj), tf.math.ceil(bj)

                    # distribute probability of Tzj
                    m_prob[i][actions[i]][int(l)] += zz_[i][next_actions[i]][j] * (u - bj)
                    m_prob[i][actions[i]][int(u)] += zz_[i][next_actions[i]][j] * (bj - l)

        m_prob = tf.stop_gradient(m_prob)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=m_prob, logits=logits)
        # loss = self.loss_dist(m_prob, logits, sample_weight=None)

        loss = tf.reduce_mean(loss)

        print(f' loss {loss}')

        td_error = 0.0
        return loss, td_error

    @tf.function
    def quantile_huber_loss(self, target, pred, actions):
        pred = tf.reduce_sum(pred * tf.expand_dims(actions, -1), axis=1)
        pred_tile = tf.tile(tf.expand_dims(pred, axis=2), [1, 1, self.atoms])
        target_tile = tf.tile(tf.expand_dims(target, axis=1), [1, self.atoms, 1])
        td_error = target_tile - pred_tile
        hub_loss = huber_loss(td_error)
        tau = tf.reshape(np.array(self.tau), [1, self.atoms])
        inv_tau = 1.0 - tau
        tau = tf.tile(tf.expand_dims(tau, axis=1), [1, self.atoms, 1])
        inv_tau = tf.tile(tf.expand_dims(inv_tau, axis=1), [1, self.atoms, 1])
        error_loss = tf.math.subtract(target_tile, pred_tile)
        loss = tf.where(tf.less(error_loss, 0.0), inv_tau * hub_loss, tau * hub_loss)
        loss = tf.reduce_mean(tf.reduce_sum(
            tf.reduce_mean(loss, axis=2), axis=1))
        return loss

    @tf.function
    def kl(self, y_target, y_pred, weights):
        a0 = y_target - tf.reduce_max(y_target, axis=-1, keepdims=True)
        a1 = y_pred - tf.reduce_max(y_pred, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.math.log(z0) - a1 + tf.math.log(z1)), axis=-1)

    @tf.function
    def _per_loss(self, y_target, y_pred):
        return tf.reduce_mean(self.is_weight * tf.math.squared_difference(y_target, y_pred))

    @tf.function
    def _kl_loss(self, y_target, y_pred, weights):  # cross_entropy loss
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_pred=y_pred, y_true=y_target, from_logits=True))

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
