import numpy as np
import tensorflow as tf

from a2c_1.utils import fc_build

from common.tf2_models import impala_fp_rnn
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
    def __init__(self, env, agent_ids, n_step, lr=0.0005, replay_buffer=None, shared_weights=False, double_q=False,
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
        self.num_agents = len(self.agent_ids)
        self.fc1_dim = 512
        self.fc2_dim = 1024
        self.rnn_unit = 256
        self.n_step = n_step
        self.one_hot_agents = tf.expand_dims(tf.one_hot(self.agent_ids, len(self.agent_ids), dtype=tf.float32), axis=1)
        print(f'self.onehot_agent is {self.one_hot_agents}')
        self.eps = tf.Variable(0., name="eps")
        self.agent = tf.constant(np.expand_dims(self.agent_ids, 1))
        print(f'self.agents shape is {self.agent.shape}')
        self.dummy_fps = np.ones((1, self.num_agents-1, self.num_actions))
        self.dummy_extra_data = np.ones((1, self.num_agents-1, 2))
        self.dummy_nstep_obs = np.ones((self.num_agents * (self.n_step-1), *self.obs_shape))
        self.dummy_done_mask = np.zeros((1, self.n_step, 1))
        print(f'tile done_mask shape is : {tf.tile(self.dummy_done_mask, (2, 1, 1)).shape}')

        with tf.name_scope('shared_q_network'):
            # self.value_network = gfootball_impala_cnn(self.obs_shape, self.num_actions, self.num_agents,
            #                               512, 512, self.shared_weights, 'shared_q_network')
            # self.value_network = impala_fp_rnn(self.obs_shape, self.fc1_dim, self.fc2_dim, 'impala_fp',
            #                                    self.num_agents, n_step=self.n_step, num_extra_data=2)
            self.value_network = impala_fp_rnn(self.obs_shape, self.fc1_dim, self.fc2_dim, self.rnn_unit, num_actions,
                                               'impala_fp_rnn', self.num_agents, self.n_step, num_extra_data=2)

        self.value_network.summary()
        tf.keras.utils.plot_model(self.value_network, to_file='./q_network_model.png')

        with tf.name_scope('shared_target_q_network'):
            # self.target_network = gfootball_impala_cnn(self.obs_shape, self.num_actions, self.num_agents,
            #                                              512, 512, self.shared_weights, 'shared_target_q_network')
            # self.target_network = impala_fp_rnn(self.obs_shape, self.fc1_dim, self.fc2_dim, 'target_impala_fp',
            #                                    self.num_agents, n_step=self.n_step, num_extra_data=2)
            self.target_network = impala_fp_rnn(self.obs_shape, self.fc1_dim, self.fc2_dim, self.rnn_unit, num_actions,
                                               'target_impala_fp_rnn', self.num_agents, self.n_step, num_extra_data=2)

        self.target_network.trainable = False

        self.q_fc_list = self._build_q_head()
        print(self.q_fc_list)
        self.target_q_fc_list = self._build_q_head()
        for target_fc in self.target_q_fc_list:
            target_fc.trainable = False

    @tf.function
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
        if self.param_noise:
            raise ValueError('not supporting noise yet')
        else:
            obs_shape = obs.shape
            output_actions = []
            fps = []
            # print(f'obs shape is {tf.concat([self.dummy_nstep_obs, obs], axis=0).shape}')
            # print(f'tf.tile(self.one_hot_agents,(self.n_step, 1 shape is {tf.tile(self.one_hot_agents,(self.n_step, 1, 1)).shape}')
            # print(f'tf.tile(self.dummy_fps, (obs_shape[0], 1, 1)) shape is {tf.tile(self.dummy_fps, (obs_shape[0]*self.n_step, 1, 1)).shape}')
            # print(f'tf.tile(self.dummy_extra_data, (obs_shape[0], 1, 1)) shape is {tf.tile(self.dummy_extra_data, (obs_shape[0]*self.n_step, 1, 1)).shape}')
            # print(f'tf.tile(self.dummy_done_mask, (obs_shape[0], 1, 1)) shape is {tf.tile(self.dummy_done_mask, (obs_shape[0], 1, 1)).shape}')

            fc_values = self.value_network({0: tf.concat([self.dummy_nstep_obs, obs], axis=0),
                                            1: tf.tile(self.one_hot_agents,(self.n_step, 1, 1)),
                                            2: tf.tile(self.dummy_fps, (obs_shape[0]*self.n_step, 1, 1)),
                                            3: tf.tile(self.dummy_extra_data, (obs_shape[0]*self.n_step, 1, 1)),
                                            4: tf.tile(self.dummy_done_mask, (obs_shape[0], 1, 1))})

            for a in self.agent_ids:
                q_values = self.q_fc_list[a](tf.expand_dims(fc_values[a], 0))
                fps.append(q_values.numpy()[0])
                deterministic_actions = tf.argmax(q_values, axis=1)

                batch_size = 1  # tf.shape(obs)[0]
                random_actions = tf.random.uniform(tf.stack([batch_size]), minval=0,
                                                   maxval=self.num_actions, dtype=tf.int64)
                chose_random = tf.random.uniform(tf.stack([batch_size]), minval=0,
                                                 maxval=1, dtype=tf.float32) < self.eps

                stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

                if stochastic:
                    output_actions.append(stochastic_actions.numpy()[0])
                else:
                    output_actions.append(deterministic_actions.numpy()[0])

            if update_eps >= 0:
                self.eps.assign(update_eps)

            return output_actions, fps

    @tf.function
    def value(self, obs):
        if self.param_noise:
            raise ValueError('not supporting noise yet')
        else:
            obs_shape = obs.shape
            fc_values = self.target_network({0: tf.concat([self.dummy_nstep_obs, obs], axis=0),
                                            1: tf.tile(self.one_hot_agents, (self.n_step, 1, 1)),
                                            2: tf.tile(self.dummy_fps, (obs_shape[0] * self.n_step, 1, 1)),
                                            3: tf.tile(self.dummy_extra_data, (obs_shape[0] * self.n_step, 1, 1)),
                                            4: tf.tile(self.dummy_done_mask, (obs_shape[0], 1, 1))})
            best_values = []
            for a in self.agent_ids:
                q_tp1 = self.target_q_fc_list[a](tf.expand_dims(fc_values[a], 0))
                q_tp1_best = tf.reduce_max(q_tp1, 1)
                best_values.append(q_tp1_best.numpy()[0])
        return best_values

    @tf.function()
    def nstep_train(self, obs0, actions, rewards, obs1, dones, importance_weights, fps, extra_datas):
        batch_size = obs0.shape[0]
        # tile_time = batch_size // self.num_agents
        td_error_ = tf.Variable(initial_value=tf.zeros(shape=batch_size//self.n_step))
        loss = tf.Variable(initial_value=0.0)
        with tf.GradientTape() as tape:
            for a in self.agent_ids:
                # print(f'obs0[:, a, :] shape is {obs0[:, a, :].shape}')
                # print(f'tf.tile(self.one_hot_agents[a], (batch_size, 1)) shape is {tf.tile(self.one_hot_agents[a], (batch_size, 1)).shape}')
                # print(f'fps[:, a, :] shape is {fps[:, a, :].shape}')
                # print(f'extra_datas[:, a, :] shape is {extra_datas[:, a, :].shape}')
                # print(f'dones shape is {dones.shape}')

                fc_values = self.value_network({0: obs0[:, a, :],
                                                1: tf.tile(self.one_hot_agents[a], (batch_size, 1)),
                                                2: fps[:, a, :],
                                                3: tf.expand_dims(extra_datas[:, a, :], axis=1),
                                                4: dones})

                q_t = self.q_fc_list[a](fc_values)
                # print(f'q_values.shape for agent {a} is {q_t.shape}')
                # print(f'tf.one_hot(actions[:, a], self.num_actions, dtype=tf.float32).shape for agent {a} is {tf.one_hot(actions[-1, a], self.num_actions, dtype=tf.float32).shape}')

                q_t_selected = tf.reduce_sum(q_t * tf.one_hot(actions[-1, a], self.num_actions, dtype=tf.float32), 1)
                # print(f'q_t_selected.dtype is {q_t_selected.dtype}')

                q_t_selected_target = rewards[-1, a]  # n-step rewards sum
                # print(f'q_t_selected_target.dtype is {q_t_selected_target.dtype}')

                td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)

                td_error_.assign_add(td_error)
                errors = huber_loss(td_error)
                weighted_error = tf.reduce_mean(importance_weights[-1, a] * errors)

                loss.assign_add(weighted_error)

        # # print(f'tile_time is {tile_time}')
        # with tf.GradientTape() as tape:
        #     fc_values = self.value_network({0: obs0,
        #                                     1: tf.tile(self.one_hot_agents, (tile_time, 1, 1)),
        #                                     2: fps,
        #                                     3: extra_datas}
        #                                    )
        #     for a in self.agent_ids:
        #
        #         q_t = self.q_fc_list[a](fc_values[])
        #     # print(f'q_values for agent {a} is {q_t}')
        #
        #     q_t_selected = tf.reduce_sum(q_t * tf.one_hot(actions, self.num_actions, dtype=tf.float32), 1)
        #     # print(f'q_t_selected.dtype is {q_t_selected.dtype}')
        #
        #     q_t_selected_target = rewards  # n-step rewards sum
        #     # print(f'q_t_selected_target.dtype is {q_t_selected_target.dtype}')
        #
        #     td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        #
        #     errors = huber_loss(td_error)
        #     weighted_error = tf.reduce_mean(importance_weights * errors)

        param = tape.watched_variables()
        grads = tape.gradient(loss, param)
        grads = [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(param, grads)]

        grads_and_vars = list(zip(grads, param))
        self.optimizer.apply_gradients(grads_and_vars)

        return td_error_

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
        if self.param_noise:
            raise ValueError('not supporting noise yet')
        else:
            obs_shape = obs_all.shape
            fc_values = self.value_network({0: tf.concat([self.dummy_nstep_obs, obs_all], axis=0),
                                            1: tf.tile(self.one_hot_agents, (self.n_step, 1, 1)),
                                            2: tf.tile(self.dummy_fps, (obs_shape[0] * self.n_step, 1, 1)),
                                            3: tf.tile(self.dummy_extra_data, (obs_shape[0] * self.n_step, 1, 1)),
                                            4: tf.tile(self.dummy_done_mask, (obs_shape[0], 1, 1))})
            output_actions = []
            for a in self.agent_ids:
                q_values = self.q_fc_list[a](tf.expand_dims(fc_values[a], 0))
                deterministic_actions = tf.argmax(q_values, axis=1)
                output_actions.append(deterministic_actions.numpy()[0])

            return output_actions

    # @tf.function(autograph=False)
    # def soft_update_target(self, local_model, target_model, tau):
    #     for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
    #         target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

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


