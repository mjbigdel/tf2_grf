import numpy as np
import tensorflow as tf

from a2c_1.utils import fc_build

from common.tf2_models import gfootball_impala_cnn
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


class MAgent(tf.Module):
    def __init__(self, env, agent_ids, lr=0.0005, replay_buffer=None, shared_weights=False, double_q=False,
                 num_actions=1, gamma=0.99, grad_norm_clipping=False, param_noise=False):
        super(MAgent, self).__init__(name='MA_DQN')
        self.env = env
        self.obs_shape = env.observation_space.shape
        self.agent_ids = agent_ids
        self.replay_buffer = replay_buffer
        self.total_reward = 0.0
        self.double_q = double_q
        self.num_actions = num_actions
        self.gamma = gamma
        self.grad_norm_clipping = grad_norm_clipping
        self.param_noise = param_noise
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.shared_weights = shared_weights

        with tf.name_scope('shared_q_network'):
            self.value_network = gfootball_impala_cnn(self.obs_shape, self.num_actions, self.agent_ids,
                                          512, 512, self.shared_weights, 'shared_q_network')
        self.value_network.summary()
        tf.keras.utils.plot_model(self.value_network, to_file='./q_network_model.png')

        with tf.name_scope('shared_target_q_network'):
            self.target_network = gfootball_impala_cnn(self.obs_shape, self.num_actions, self.agent_ids,
                                                         512, 512, self.shared_weights, 'shared_target_q_network')
        self.target_network.trainable = False
        self.target_network.summary()

        self.q_fc_list = self._build_q_head()
        print(self.q_fc_list)
        self.target_q_fc_list = self._build_q_head()
        for target_fc in self.target_q_fc_list:
            target_fc.trainable = False

        self.eps = tf.Variable(0., name="eps")

        self.agent = tf.constant(np.expand_dims(self.agent_ids, 1))
        print(f'self.agents shape is {self.agent.shape}')

    def _build_q_head(self):
        input_shape = self.value_network.output_shape
        name = 'q'
        critics_fc = []
        if self.agent_ids is not None:
            name += '_agent' + str(self.agent_ids)
        for a in self.agent_ids:
            name += '_' + str(a)
            critics_fc.append(fc_build(input_shape, name, self.num_actions))
        return critics_fc

    @tf.function
    def choose_action(self, obs, stochastic=True, update_eps=-1):
        output_actions = []
        if self.param_noise:
            raise ValueError('not supporting noise yet')
        else:
            fc_values = self.value_network({0: obs, 1: self.agent})
            # print(f'fc_values is {fc_values}')
            for a in self.agent_ids:
                q_values = self.q_fc_list[a](tf.expand_dims(fc_values[a], 0))
                # print('====================================')

                # print(f'q_values is {q_values}')
                deterministic_actions = tf.argmax(q_values, axis=1)
                batch_size = 1  # tf.shape(obs)[0]
                random_actions = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=self.num_actions,
                                               dtype=tf.int64)
                chose_random = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < self.eps
                stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

                if stochastic:
                    output_actions.append(stochastic_actions.numpy()[0])
                else:
                    output_actions.append(deterministic_actions.numpy()[0])

            if update_eps >= 0:
                self.eps.assign(update_eps)

            return output_actions, fc_values

    @tf.function()
    def train(self, obs0, actions, rewards, obs1, dones, importance_weights, fps, extra_datas):
        batch_size = obs0.shape[0]
        td_error_ = tf.Variable(initial_value=tf.zeros(shape=batch_size))
        for a in self.agent_ids:
            with tf.GradientTape() as tape:
                fc_values = self.value_network({0: obs0[:, a, :], 1: tf.ones(shape=(batch_size, 1)) * a})
                q_t = self.q_fc_list[a](fc_values)

                q_t_selected = tf.reduce_sum(q_t * tf.one_hot(actions[:, a], self.num_actions, dtype=tf.float32), 1)

                fc_tp1 = self.target_network({0: obs1[:, a, :], 1: tf.ones(shape=(batch_size, 1)) * a})
                q_tp1 = self.target_q_fc_list[a](fc_tp1)

                if self.double_q:
                    fc_tp1_using_online_net = self.value_network({0: obs1[:, a, :], 1: tf.ones(shape=(batch_size, 1)) * a})
                    q_tp1_using_online_net = self.q_fc_list[a](fc_tp1_using_online_net)
                    q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
                    q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, self.num_actions, dtype=tf.float32), 1)
                else:
                    q_tp1_best = tf.reduce_max(q_tp1, 1)

                dones = tf.cast(dones, q_tp1_best.dtype)
                q_tp1_best_masked = (1.0 - dones) * q_tp1_best

                q_t_selected_target = rewards[:, a] + self.gamma * q_tp1_best_masked

                td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
                td_error_.assign_add(td_error)
                errors = huber_loss(td_error)
                weighted_error = tf.reduce_mean(importance_weights[:, a] * errors)

                # loss.assign_add(weighted_error)

                param = tape.watched_variables()

                # print(param)
                # param = [v for v in self.q_network.trainable_variables if v.name.__contains__(agent_name)]
                # param += [v for v in self.q_network.trainable_variables if not v.name.__contains__('agent')]
                # print(f'params for is {param}')

                grads = tape.gradient(weighted_error, param)
                grads = [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(param, grads)]

                grads_and_vars = list(zip(grads, param))
                self.optimizer.apply_gradients(grads_and_vars)

        return td_error

    def choose_greedy_action(self, obs_all):
        output_actions = []
        if self.param_noise:
            raise ValueError('not supporting noise yet')
        else:
            fc_values = self.value_network({0: obs_all, 1: self.agent})
            for a in self.agent_ids:
                q_values = self.q_fc_list[a](tf.expand_dims(fc_values[a], 0))
                deterministic_actions = tf.argmax(q_values, axis=1)
                output_actions.append(deterministic_actions.numpy()[0])

            return output_actions

    @tf.function(autograph=False)
    def update_target(self):
        q_vars = self.value_network.trainable_variables
        target_q_vars = self.target_network.trainable_variables
        for var, var_target in zip(q_vars, target_q_vars):
            var_target.assign(var)

        q_fc_vars = []
        target_q_fc_vars = []
        for a in self.agent_ids:
            q_fc_vars.extend(self.q_fc_list[a].trainable_variables)
            target_q_fc_vars.extend(self.target_q_fc_list[a].trainable_variables)

        for var_, var_target_ in zip(q_fc_vars, target_q_fc_vars):
            var_target_.assign(var_)


