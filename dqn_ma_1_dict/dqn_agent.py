import numpy as np
import tensorflow as tf

from dqn_ma_1_dict.tf2_models import build_ma_mlp, build_ma_cnn, build_cnn, build_mlp
from common.tf2_utils import huber_loss


class Agent:
    def __init__(self, env, replay_buffer, epsilon=0.0):
        self.env = env
        self.replay_buffer = replay_buffer
        self.obses_t = env.reset()
        self.total_reward = 0.0
        self.network = None
        self.epsilon = epsilon

    def choose_action(self, obs):
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            tmp = self.network.learning_network(obs)
            action = tf.math.argmax(tmp[0])
        return action

    def play_step(self):
        done_reward = None
        # select and action
        action = self.choose_action(self.obses_t)

        # feed the action to the Environment
        obses_tp1, reward, done, _ = self.env.step(action)
        self.total_reward += reward

        # add the Experience to buffer
        self.replay_buffer.add(self.obses_t, action, reward, obses_tp1, float(done))

        # update obses_t to new obs
        self.obses_t = obses_tp1

        # check if game is terminated by done
        if done:
            done_reward = self.total_reward
            self.obses_t = self.env.reset()
            self.total_reward = 0.0

        return done_reward


class DeepQAgent:
    def __init__(self, env, agent_names, lr=0.0005, replay_buffer=None, double_q=False,
                 num_actions=1, gamma=0.99, grad_norm_clipping=False, param_noise=False):
        self.env = env
        self.agent_names = agent_names
        self.input_shape = env.observation_space.shape
        print(self.input_shape)
        self.replay_buffer = replay_buffer
        self.obses_t = env.reset()
        self.total_reward = 0.0
        self.double_q = double_q
        self.num_actions = num_actions
        self.gamma = gamma
        self.grad_norm_clipping = grad_norm_clipping
        self.optimizer = tf.keras.optimizers.Adam(lr)

        self.param_noise = param_noise
        with tf.name_scope('q_network'):
            self.q_network = build_cnn(self.input_shape, num_actions, fc1_dims=512)
        with tf.name_scope('target_q_network'):
            self.target_q_network = build_cnn(self.input_shape, num_actions, fc1_dims=512)
        self.eps = tf.Variable(0., name="eps")

    # def choose_action(self, obs):
    #     if np.random.random() < self.epsilon:
    #         action = self.env.action_space.sample()
    #     else:
    #         tmp = self.q_network(obs)
    #         action = tf.math.argmax(tmp[0])
    #     return action

    @tf.function
    def choose_action(self, obs, stochastic=True, update_eps=-1):
        if self.param_noise:
            raise ValueError('not supporting noise yet')
        else:
            actions_dict = {}
            for agent_name in self.agent_names:
                q_values = self.q_network(obs[agent_name])
                deterministic_actions = tf.argmax(q_values, axis=1)
                batch_size = tf.shape(obs[agent_name])[0]
                random_actions = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=self.num_actions,
                                               dtype=tf.int64)
                chose_random = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < self.eps
                stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

                if stochastic:
                    output_actions = stochastic_actions
                else:
                    output_actions = deterministic_actions


            if update_eps >= 0:
                self.eps.assign(update_eps)

            return output_actions

    def play_step(self):
        done_reward = None
        # select and action
        output_actions = self.choose_action(self.obses_t)

        # feed the action to the Environment
        obses_tp1, reward, done, _ = self.env.step(output_actions)
        self.total_reward += reward

        if self.replay_buffer is not None:
            # add the Experience to buffer
            self.replay_buffer.add(self.obses_t, output_actions, reward, obses_tp1, float(done))

        # update obses_t to new obs
        self.obses_t = obses_tp1

        # check if game is terminated by done
        if done:
            done_reward = self.total_reward
            self.obses_t = self.env.reset()
            self.total_reward = 0.0

        return done_reward

    @tf.function()
    def train(self, obs0, actions, rewards, obs1, dones, importance_weights):
        with tf.GradientTape() as tape:
            q_t = self.q_network(obs0)
            q_t_selected = tf.reduce_sum(q_t * tf.one_hot(actions, self.num_actions, dtype=tf.float32), 1)

            q_tp1 = self.target_q_network(obs1)

            if self.double_q:
                q_tp1_using_online_net = self.q_network(obs1)
                q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
                q_tp1_best = tf.reduce_sum(
                    q_tp1 * tf.one_hot(q_tp1_best_using_online_net, self.num_actions, dtype=tf.float32), 1)
            else:
                q_tp1_best = tf.reduce_max(q_tp1, 1)

            dones = tf.cast(dones, q_tp1_best.dtype)
            q_tp1_best_masked = (1.0 - dones) * q_tp1_best

            q_t_selected_target = rewards + self.gamma * q_tp1_best_masked

            td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
            errors = huber_loss(td_error)
            weighted_error = tf.reduce_mean(importance_weights * errors)

        grads = tape.gradient(weighted_error, self.q_network.trainable_variables)
        if self.grad_norm_clipping:
            clipped_grads = []
            for grad in grads:
                clipped_grads.append(tf.clip_by_norm(grad, self.grad_norm_clipping))
            clipped_grads = grads
        grads_and_vars = zip(grads, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(grads_and_vars)

        return td_error

    @tf.function(autograph=False)
    def update_target(self):
        q_vars = self.q_network.trainable_variables
        target_q_vars = self.target_q_network.trainable_variables
        for var, var_target in zip(q_vars, target_q_vars):
            var_target.assign(var)


class MAgent:
    def __init__(self, env, agent_names, lr=0.0005, replay_buffer=None, shared_weights=False, double_q=False,
                 num_actions=1, gamma=0.99, grad_norm_clipping=False, param_noise=False):
        self.env = env
        self.obs_shape = env.observation_space.shape
        self.agent_names = agent_names
        self.replay_buffer = replay_buffer
        # self.obses_t = env.reset()
        self.total_reward = 0.0
        self.double_q = double_q
        self.num_actions = num_actions
        self.gamma = gamma
        self.grad_norm_clipping = grad_norm_clipping
        self.param_noise = param_noise
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.shared_weights = shared_weights

        with tf.name_scope('shared_q_network'):
            self.q_network = build_ma_cnn(self.obs_shape, self.num_actions, self.agent_names,
                                          512, 512, self.shared_weights, 'shared_q_network')
        self.q_network.compile(self.optimizer)
        self.q_network.summary()

        with tf.name_scope('shared_target_q_network'):
            self.target_q_network = build_ma_cnn(self.obs_shape, self.num_actions, self.agent_names,
                                                 512, 512, self.shared_weights, 'shared_target_q_network')
        self.target_q_network.compile(self.optimizer)
        self.target_q_network.summary()
        self.eps = tf.Variable(0., name="eps")

    # @tf.function
    def choose_action(self, obs_dict, stochastic=True, update_eps=-1):
        if self.param_noise:
            raise ValueError('not supporting noise yet')
        else:
            actions_dict = {}
            for agent_name in self.agent_names:
                # obs = tf.reshape(obs_dict[agent_name], (1, *self.obs_shape))
                obs = np.expand_dims(obs_dict[agent_name], axis=0)
                # print(f'obs.shape {obs.shape}')  # Tensor("Reshape:0", shape=(1, 42, 42, 3), dtype=uint8)
                q_values = self.q_network(obs)[agent_name]
                # print(f'q_values {q_values}')  # Tensor("shared_q_network/output_agent_0/BiasAdd:0", shape=(1, 19), dtype=float32)
                deterministic_actions = tf.argmax(q_values, axis=1)
                batch_size = tf.shape(obs)[0]
                random_actions = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=self.num_actions,
                                                   dtype=tf.int64)
                chose_random = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < self.eps
                stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

                if stochastic:
                    output_actions = stochastic_actions
                else:
                    output_actions = deterministic_actions

                # print(output_actions[0].numpy())

                actions_dict[agent_name] = output_actions[0].numpy()

            if update_eps >= 0:
                self.eps.assign(update_eps)

            return actions_dict

    @tf.function()
    def train(self, obs0, actions, rewards, obs1, dones, importance_weights, agent_name):
        with tf.GradientTape() as tape:
            q_t = self.q_network(obs0)[agent_name]
            q_t_selected = tf.reduce_sum(q_t * tf.one_hot(actions, self.num_actions, dtype=tf.float32), 1)

            q_tp1 = self.target_q_network(obs1)[agent_name]

            if self.double_q:
                q_tp1_using_online_net = self.q_network(obs1)[agent_name]
                q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
                q_tp1_best = tf.reduce_sum(
                    q_tp1 * tf.one_hot(q_tp1_best_using_online_net, self.num_actions, dtype=tf.float32), 1)
            else:
                q_tp1_best = tf.reduce_max(q_tp1, 1)

            dones = tf.cast(dones, q_tp1_best.dtype)
            q_tp1_best_masked = (1.0 - dones) * q_tp1_best

            q_t_selected_target = rewards + self.gamma * q_tp1_best_masked

            td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
            errors = huber_loss(td_error)
            weighted_error = tf.reduce_mean(importance_weights * errors)

        param = [v for v in self.q_network.trainable_variables if v.name.__contains__(agent_name)]
        grads = tape.gradient(weighted_error, param)
        if self.grad_norm_clipping:
            clipped_grads = []
            for grad in grads:
                clipped_grads.append(tf.clip_by_norm(grad, self.grad_norm_clipping))
            clipped_grads = grads
        grads_and_vars = zip(grads, param)
        self.optimizer.apply_gradients(grads_and_vars)

        return td_error

    def choose_greedy_action(self, obs_dict):
        if self.param_noise:
            raise ValueError('not supporting noise yet')
        else:
            actions_dict = {}
            for agent_name in self.agent_names:
                # obs = tf.reshape(obs_dict[agent_name], (1, *self.obs_shape))
                obs = np.expand_dims(obs_dict[agent_name], axis=0)
                # print(f'obs.shape {obs.shape}')  # Tensor("Reshape:0", shape=(1, 42, 42, 3), dtype=uint8)
                q_values = self.q_network(obs)[agent_name]
                # print(f'q_values {q_values}')  # Tensor("shared_q_network/output_agent_0/BiasAdd:0", shape=(1, 19), dtype=float32)
                deterministic_actions = tf.argmax(q_values, axis=1)
                # batch_size = tf.shape(obs)[0]
                # random_actions = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=self.num_actions,
                #                                    dtype=tf.int64)
                # chose_random = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < self.eps
                # stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

                # if stochastic:
                #     output_actions = stochastic_actions
                # else:
                #     output_actions = deterministic_actions

                # print(output_actions[0].numpy())

                actions_dict[agent_name] = deterministic_actions[0].numpy()

            # if update_eps >= 0:
            #     self.eps.assign(update_eps)

            return actions_dict

    @tf.function(autograph=False)
    def update_target(self):
        q_vars = self.q_network.trainable_variables
        target_q_vars = self.target_q_network.trainable_variables
        for var, var_target in zip(q_vars, target_q_vars):
            var_target.assign(var)
