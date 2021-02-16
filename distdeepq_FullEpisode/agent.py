import numpy as np
import tensorflow as tf
from distdeepq_FullEpisode.utils import huber_loss


# Agent
class Agent:
    def __init__(self, config, model, target_model, agent_id):
        super().__init__()
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

        self.dummy_done = tf.zeros((1, 1))
        self.dummy_fps = tf.zeros((1, 1, config.fp_shape))

    @tf.function
    def max_value(self, obs):
        """
        :param obs: list observations one for each agent
        :return: best values based on Q-Learning formula maxQ(s',a')
        """
        obs = tf.expand_dims(obs, axis=0)
        if self.config.is_recurrent:
            inputs = [tf.expand_dims(obs, axis=0), self.dummy_done, self.dummy_fps]
        else:
            inputs = obs

        q_tp1 = self.target_model(inputs)
        # print(f' q_tp1 {q_tp1}')

        if self.config.double_q:
            q_values_using_online_net = self.model(inputs)
            q_value_best_using_online_net = tf.argmax(q_values_using_online_net, 1)
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
        if self.config.is_recurrent:
            inputs = [tf.expand_dims(obs, axis=0), self.dummy_done, self.dummy_fps]
        else:
            inputs = obs

        # print(f' obs.shape {obs.shape}')
        zz = self.target_model(inputs)
        # print(f' zz.shape {zz.shape}')
        # z_concat = np.vstack(zz)
        # print(f' z_concat.shape {z_concat.shape}')
        q_tp1 = tf.reduce_sum(tf.math.multiply(zz, self.z), axis=-1)
        # q_tp1 = self.target_model(obs)
        # print(f' q_tp1.shape {q_tp1.shape}')

        if self.config.double_q:
            z_using_online_net = self.model(inputs)
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
        if self.config.is_recurrent:
            inputs = [tf.expand_dims(obs, axis=0), self.dummy_done, self.dummy_fps]
        else:
            inputs = obs

        q_values = self.model(inputs)
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

        if self.config.is_recurrent:
            inputs = [tf.expand_dims(obs, axis=0), self.dummy_done, self.dummy_fps]
        else:
            inputs = obs


        zz = self.model(inputs)  # shape (1, 19, 8)
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
        if self.config.is_recurrent:
            inputs = [obses_t, dones, fps]
        else:
            inputs = obses_t

        q_t = self.model(inputs)

        rewards = rewards[:, 0]
        actions = actions[:, 0]
        dones = dones[:, -1]

        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(actions, self.config.num_actions, dtype=tf.float32), 1)
        # print(f'q_t_selected.shape is {q_t_selected.shape}')

        td_error = q_t_selected - tf.stop_gradient(rewards)

        errors = huber_loss(td_error)
        weighted_loss = tf.reduce_mean(weights * errors)

        return weighted_loss, td_error

    @tf.function()
    def compute_loss_dist(self, obses_t, actions, rewards, obs_tp1, dones, weights, fps=None):
        # print(f' obses_t.shape {obses_t.shape}')
        # print(f' dones.shape {dones.shape}')
        if self.config.is_recurrent:
            inputs = [obses_t, dones[:, :-1], fps]
        else:
            inputs = obses_t[:, 0, :]

        logits = self.model(inputs)
        # print(f' logits.shape {logits.shape}')

        obses_t = obses_t[:, 0, :]
        rewards = rewards[:, 0]
        actions = actions[:, 0]
        dones = dones[:, -1]  # done for obs_tp1
        # fps = fps[:, -1]  # fps for obs_tp1

        # print(f' obses_t.shape {obses_t.shape}')

        m_prob = tf.stop_gradient(self.build_target_ditribution(obses_t, actions,
                                                                rewards, obs_tp1, dones, weights, fps=None))

        # print(f' m_prob.shape {np.array(m_prob).shape}')

        # loss = tf.nn.softmax_cross_entropy_with_logits(labels=m_prob, logits=logits)
        # loss = tf.reduce_mean(loss)
        # td_error = 1.0

        # print(f' loss {loss}')
        td_error = logits - m_prob
        # print(f' td_errors {td_error}')
        td_error = tf.reduce_sum(td_error, axis=[1, 2])
        # print(f' td_errors {td_error}')
        errors = huber_loss(td_error)
        # print(f' errors {errors}')
        weighted_loss = tf.reduce_mean(weights * errors)
        # print(f' weighted_loss {weighted_loss}')
        loss = weighted_loss

        # print(f' td_errors {td_error}')

        return loss, td_error

    def build_target_ditribution(self, obses_t, actions, rewards, obs_tp1, dones, weights, fps=None):
        if self.config.is_recurrent:
            inputs = [tf.expand_dims(obs_tp1, axis=1), tf.expand_dims(dones, axis=1),
                      tf.tile(self.dummy_fps, (self.config.batch_size, 1, 1))]
        else:
            inputs = obs_tp1

        zz = self.model(inputs)
        # print(f' zz.shape {zz.shape}')
        q = tf.reduce_sum(tf.math.multiply(zz, self.z), axis=-1)
        # print(f' q.shape {q.shape}')
        next_actions = tf.argmax(q, axis=1)  # a* in C51 algo

        zz_ = self.target_model(inputs)
        # zz_ = tf.stop_gradient(zz_)
        # rewards = tf.stop_gradient(rewards)
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

        return m_prob

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
