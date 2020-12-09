import tensorflow as tf
from common.tf2_utils import huber_loss



class Agent(tf.Module):
    def __init__(self, q_func, obs_shape, agent_ids, lr=0.0005, tau=0.01, double_q=False,
                 num_actions=1, gamma=0.99, grad_norm_clipping=False, param_noise=False):
        super(Agent, self).__init__(name='nstep_dqn')
        self.obs_shape = obs_shape
        self.agent_ids = agent_ids
        self.double_q = double_q
        self.num_actions = num_actions
        self.gamma = gamma
        self.grad_norm_clipping = grad_norm_clipping
        self.param_noise = param_noise
        self.tau = tau
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.one_hot_agents = tf.expand_dims(tf.one_hot(self.agent_ids, len(self.agent_ids), dtype=tf.float32), axis=1)

        with tf.name_scope('q_network'):
            self.q_network = q_func(obs_shape, num_actions, agent_ids)
        with tf.name_scope('target_q_network'):
            self.target_q_network = q_func(obs_shape, num_actions, agent_ids)
        self.eps = tf.Variable(0., name="eps")

        self.q_network.summary()
        tf.keras.utils.plot_model(self.q_network, to_file='./q_network.png')
        print(f'self.q_network.outputs is {self.q_network.outputs}')

    @tf.function
    def choose_action(self, obs, stochastic=True, update_eps=-1):
        output_actions = []
        if self.param_noise:
            raise ValueError('not supporting noise yet')
        else:
            q_values = self.q_network({0: tf.expand_dims(obs[0], 0), 1: self.one_hot_agents[0]})[0]
            # print(f'q_values is {q_values}')
            deterministic_action = tf.argmax(q_values)

            print(f'self.eps is {self.eps} rand is {tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32)}')
            if tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32) < self.eps:
                random_action = tf.random.uniform([], minval=0, maxval=self.num_actions, dtype=tf.int64)
                stochastic_action = random_action
                print(f'random_action is {random_action}')
            else:
                stochastic_action = deterministic_action
                print(f'deterministic_action is {deterministic_action}')

            output_actions.append(stochastic_action.numpy())

        if update_eps >= 0:
            self.eps.assign(update_eps)

        print(f'output_actions is {output_actions}')
        return output_actions

    def choose_greedy_action(self, obs):
        output_actions = []
        for a in self.agent_ids:
            q_values = self.q_network({0: tf.expand_dims(obs[0], 0), 1: self.one_hot_agents[0]})[0]
            deterministic_action = tf.argmax(q_values)

            output_actions.append(deterministic_action.numpy())

        print(f'output_actions is {output_actions}')
        return output_actions

    @tf.function
    def value(self, obs):
        q_values_ = []
        if self.param_noise:
            raise ValueError('not supporting noise yet')
        else:
            q_tp1 = self.target_q_network({0: tf.expand_dims(obs[0], 0), 1: self.one_hot_agents[0]})[0]
            if self.double_q:
                q_tp1_using_online_net = self.q_network({0: tf.expand_dims(obs[0], 0), 1: self.one_hot_agents[0]})[0]
                q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net)
                q_tp1_best = tf.reduce_sum(
                    q_tp1 * tf.one_hot(q_tp1_best_using_online_net, self.num_actions, dtype=tf.float32))
            else:
                q_tp1_best = tf.reduce_max(q_tp1)

            q_values_.append(q_tp1_best.numpy())

        print(f'qvalues is {q_values_}')
        return q_values_

    @tf.function()
    def nstep_train(self, obs0, actions, rewards, obs1, dones, importance_weights, fps, extra_datas):
        batch_size = obs0.shape[0]
        td_error_ = []
        loss = []
        with tf.GradientTape() as tape:
            for a in self.agent_ids:
                q_t = self.q_network({0: obs0[:, a, :], 1: tf.tile(self.one_hot_agents[a], (batch_size, 1))})[a]
                # print(f'q_values for agent {a} is {q_t}')

                q_t_selected = tf.reduce_sum(q_t * tf.one_hot(actions[:, a], self.num_actions, dtype=tf.float32), 1)
                # print(f'q_t_selected is {q_t_selected}')

                q_t_selected_target = rewards[:, a]  # n-step rewards sum
                # print(f'q_t_selected_target is {q_t_selected_target}')

                td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)

                td_error_.append(td_error)
                errors = huber_loss(td_error)
                weighted_error = tf.reduce_mean(importance_weights[:, a] * errors)

                loss.append(weighted_error)

            sum_loss = tf.reduce_sum(loss)
            sum_td_error = tf.reduce_sum(td_error_)

        print(f'sum_loss is {sum_loss}, loss is {loss}')
        param = self.q_network.trainable_variables
        grads = tape.gradient(sum_loss, param)
        if self.grad_norm_clipping:
            clipped_grads = []
            for grad in grads:
                clipped_grads.append(tf.clip_by_norm(grad, self.grad_norm_clipping))
            grads = clipped_grads

        grads_and_vars = list(zip(grads, param))
        self.optimizer.apply_gradients(grads_and_vars)

        return sum_loss.numpy(), sum_td_error.numpy()

    @tf.function(autograph=False)
    def update_target(self):
        for var, var_target in zip(self.q_network.trainable_variables, self.target_q_network.trainable_variables):
            var_target.assign(var)

    @tf.function(autograph=False)
    def soft_update_target(self):
        for var, var_target in zip(self.q_network.trainable_variables, self.target_q_network.trainable_variables):
            var_target.assign(self.tau * var + (1.0 - self.tau) * var_target)



class MAgent(tf.Module):
    def __init__(self, q_func, obs_shape, agent_ids, lr=0.0005, tau=0.01, double_q=False,
                 num_actions=1, gamma=0.99, grad_norm_clipping=False, param_noise=False):
        super(MAgent, self).__init__(name='nstep_ma_dqn')
        self.obs_shape = obs_shape
        self.agent_ids = agent_ids
        self.double_q = double_q
        self.num_actions = num_actions
        self.gamma = gamma
        self.grad_norm_clipping = grad_norm_clipping
        self.param_noise = param_noise
        self.tau = tau
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.one_hot_agents = tf.expand_dims(tf.one_hot(self.agent_ids, len(self.agent_ids), dtype=tf.float32), axis=1)
        print(f'self.one_hot_agents is {self.one_hot_agents}')

        with tf.name_scope('q_network'):
            self.q_network = q_func(obs_shape, num_actions, agent_ids)
        with tf.name_scope('target_q_network'):
            self.target_q_network = q_func(obs_shape, num_actions, agent_ids)
        self.eps = tf.Variable(0., name="eps")

        self.q_network.summary()
        tf.keras.utils.plot_model(self.q_network, to_file='./q_network.png')
        print(f'self.q_network.outputs is {self.q_network.outputs}')

    @tf.function
    def choose_action(self, obs, stochastic=True, update_eps=-1):
        output_actions = []
        if self.param_noise:
            raise ValueError('not supporting noise yet')
        else:
            for a in self.agent_ids:
                print(tf.shape(obs[a]))
                q_values = self.q_network({0: tf.expand_dims(obs[a], 0), 1: self.one_hot_agents[a]})[a]
                # print(f'q_values is {q_values}')
                deterministic_action = tf.argmax(q_values, -1)

                print(f'self.eps is {self.eps} rand is {tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32)}')
                if tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32) < self.eps:
                    random_action = tf.random.uniform([1], minval=0, maxval=self.num_actions, dtype=tf.int64)
                    stochastic_action = random_action
                    print(f'random_action is {random_action}')
                else:
                    stochastic_action = deterministic_action
                    print(f'deterministic_action is {deterministic_action}')

                output_actions.append(stochastic_action.numpy()[0])

            if update_eps >= 0:
                self.eps.assign(update_eps)

            print(f'output_actions is {output_actions}')
            return output_actions

    def choose_greedy_action(self, obs):
        output_actions = []
        for a in self.agent_ids:
            q_values = self.q_network({0: tf.expand_dims(obs[a], 0), 1: self.one_hot_agents[a]})[a]
            deterministic_action = tf.argmax(q_values, -1)

            output_actions.append(deterministic_action.numpy()[0])

        print(f'output_actions is {output_actions}')
        return output_actions

    @tf.function
    def value(self, obs):
        q_values_ = []
        if self.param_noise:
            raise ValueError('not supporting noise yet')
        else:
            for a in self.agent_ids:
                q_tp1 = self.target_q_network({0: tf.expand_dims(obs[a], 0), 1: self.one_hot_agents[a]})[a]

                if self.double_q:
                    q_tp1_using_online_net = self.q_network({0: tf.expand_dims(obs[a], 0), 1: self.one_hot_agents[a]})[a]
                    q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, -1)
                    q_tp1_best = tf.reduce_sum(
                        q_tp1 * tf.one_hot(q_tp1_best_using_online_net, self.num_actions, dtype=tf.float32), -1)
                else:
                    q_tp1_best = tf.reduce_max(q_tp1, -1)


                q_values_.append(q_tp1_best.numpy()[0])

        print(f'q_values_ is {q_values_}')

        return q_values_

    @tf.function()
    def nstep_train(self, obs0, actions, rewards, obs1, dones, importance_weights, fps, extra_datas):
        batch_size = obs0.shape[0]
        td_error_ = []
        loss = []
        with tf.GradientTape() as tape:
            for a in self.agent_ids:
                q_t = self.q_network({0: obs0[:, a, :], 1: tf.tile(self.one_hot_agents[a], (batch_size, 1))})[a]
                # print(f'q_values for agent {a} is {q_t}')

                q_t_selected = tf.reduce_sum(q_t * tf.one_hot(actions[:, a], self.num_actions, dtype=tf.float32), -1)
                # print(f'q_t_selected is {q_t_selected}')

                q_t_selected_target = rewards[:, a]  # n-step rewards sum
                # print(f'q_t_selected_target is {q_t_selected_target}')

                td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)

                td_error_.append(td_error)
                errors = huber_loss(td_error)
                weighted_error = tf.reduce_mean(importance_weights[:, a] * errors)

                loss.append(weighted_error)

            sum_loss = tf.reduce_sum(loss)
            sum_td_error = tf.reduce_sum(td_error_)

        print(f'sum_loss is {sum_loss}, loss is {loss}')
        param = self.q_network.trainable_variables
        grads = tape.gradient(sum_loss, param)
        if self.grad_norm_clipping:
            clipped_grads = []
            for grad in grads:
                clipped_grads.append(tf.clip_by_norm(grad, self.grad_norm_clipping))
            grads = clipped_grads

        grads_and_vars = list(zip(grads, param))
        self.optimizer.apply_gradients(grads_and_vars)

        return sum_loss.numpy(), sum_td_error.numpy()

    @tf.function(autograph=False)
    def update_target(self):
        for var, var_target in zip(self.q_network.trainable_variables, self.target_q_network.trainable_variables):
            var_target.assign(var)

    @tf.function(autograph=False)
    def soft_update_target(self):
        for var, var_target in zip(self.q_network.trainable_variables, self.target_q_network.trainable_variables):
            var_target.assign(self.tau * var + (1.0 - self.tau) * var_target)

