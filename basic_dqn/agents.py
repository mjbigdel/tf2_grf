from _ast import mod

import tensorflow as tf
import numpy as np
import time

from common.schedules import LinearSchedule
from common import logger
from common.tf2_utils import huber_loss
from common.utils import init_env

from basic_dqn.utils import init_replay_memory, init_network


class Agent(tf.Module):
    def __init__(self, config, env):
        self.config = config
        self.agent_ids = [a for a in range(config.num_agents)]
        self.env = env
        self.optimizer = tf.keras.optimizers.Adam(self.config.lr)
        self.replay_memory, self.beta_schedule = init_replay_memory(config)

        self.model = init_network(config)
        self.target_model = init_network(config)
        self.model.summary()
        tf.keras.utils.plot_model(self.model, to_file='./model.png')

        if self.config.dueling:
            self.agent_heads = self.build_agent_heads_dueling()
            self.target_agent_heads = self.build_agent_heads_dueling()
            self.agent_heads[0].summary()
            tf.keras.utils.plot_model(self.agent_heads[0], to_file='./agent_heads_model.png')
        else:
            self.agent_heads = self.build_agent_heads()
            self.target_agent_heads = self.build_agent_heads()

        # Create the schedule for exploration starting from 1.
        self.exploration = LinearSchedule(schedule_timesteps=int(config.exploration_fraction * config.num_timesteps),
                                          initial_p=1.0, final_p=config.exploration_final_eps)

        if config.load_path is not None:
            self.load_models(config.load_path)

        self.loss = self.nstep_loss
        self.eps = tf.Variable(0.0)
        self.one_hot_agents = tf.expand_dims(tf.one_hot(self.agent_ids, len(self.agent_ids), dtype=tf.float32), axis=1)
        print(f'self.onehot_agent.shape is {self.one_hot_agents.shape}')

    def build_agent_heads(self):
        """

        :return: list of heads for agents

            - gets tensorflow model and adds heads for each agent
        """
        input_shape = self.model.output_shape
        heads = []
        for a in self.agent_ids:
            name = 'head_agent_' + str(a)
            head_a = tf.keras.layers.Dense(units=self.config.num_actions, activation=None,
                                           kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                           bias_initializer=tf.keras.initializers.Constant(0.0),
                                           name=name)
            head_a.build(input_shape)
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
                state_head_a = tf.keras.layers.Dense(units=self.config.num_actions, activation=None,
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
    def choose_action(self, obs, stochastic=True, update_eps=-1):
        """

        :param obs: list observations one for each agent
        :param stochastic: True for Train phase and False for test phase
        :param update_eps: epsilon update for eps-greedy
        :return: actions: list of actions chosen by agents based on observation one for each agent
        """

        actions = []
        for a in self.agent_ids:
            inputs = {0: np.expand_dims(obs[a], 0), 1: self.one_hot_agents[a]}
            fc_values = self.model(inputs)
            # print(f'fc_values.shape {fc_values.shape}')
            q_values = self.agent_heads[a](fc_values)
            # print(f'q_values.shape {q_values.shape}')
            deterministic_actions = tf.argmax(q_values, axis=1)
            # print(f'deterministic_actions {deterministic_actions}')

            batch_size = 1
            random_actions = tf.random.uniform(tf.stack([batch_size]), minval=0,
                                               maxval=self.config.num_actions, dtype=tf.int64)
            # print(f'random_actions {random_actions}')
            chose_random = tf.random.uniform(tf.stack([batch_size]), minval=0,
                                             maxval=1, dtype=tf.float32) < self.eps
            # print(f'chose_random {chose_random}')

            stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)
            # print(f'stochastic_actions {stochastic_actions}')

            if stochastic:
                actions.append(stochastic_actions.numpy()[0])
            else:
                actions.append(deterministic_actions.numpy()[0])

        if update_eps >= 0:
            self.eps.assign(update_eps)

        # print(f'actions {actions}')
        return actions

    @tf.function
    def value(self, obs):
        """

        :param obs: list observations one for each agent
        :return: best values based on Q-Learning formula max Q(s',a')
        """

        values = []
        for a in self.agent_ids:
            inputs = {0: np.expand_dims(obs[a], 0), 1: self.one_hot_agents[a]}
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

    @tf.function()
    def nstep_loss(self, obses_t, actions, rewards, weights, agent_id):
        # print(f'obses_t.shape {obses_t.shape}')
        s = obses_t.shape
        obses_t = tf.reshape(obses_t, (s[0]*s[1], *s[2:]))
        # print(f'obses_t.shape {obses_t.shape}')
        s = actions.shape
        actions = tf.reshape(actions, (s[0] * s[1], *s[2:]))
        # print(f'actions.shape {actions.shape}')
        s = rewards.shape
        rewards = tf.reshape(rewards, (s[0] * s[1], *s[2:]))
        # print(f'rewards.shape {rewards.shape}')
        s = weights.shape
        weights = tf.reshape(weights, (s[0] * s[1], *s[2:]))
        # print(f'weights.shape {weights.shape}')

        inputs = {0: obses_t, 1: tf.tile(self.one_hot_agents[agent_id], (s[0]*s[1], 1))}
        fc_values = self.model(inputs)
        q_t = self.agent_heads[agent_id](fc_values)

        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(actions, self.config.num_actions, dtype=tf.float32), 1)
        # print(f'q_t_selected.shape is {q_t_selected.shape}')

        td_error = q_t_selected - tf.stop_gradient(rewards)

        errors = huber_loss(td_error)
        weighted_loss = tf.reduce_mean(weights * errors)

        return weighted_loss, td_error

    @tf.function()
    def train(self, obses_t, actions, rewards, weights):
        # print(f'obses_t.shape {obses_t.shape}')
        td_errors = []
        loss = []
        with tf.GradientTape() as tape:
            for a in self.agent_ids:
                loss_a, td_error = self.loss(obses_t[:, a], actions[:, a], rewards[:, a], weights[:, a], a)
                loss.append(loss_a)
                td_errors.append(td_error)

            sum_loss = tf.reduce_sum(loss)
            sum_td_error = tf.reduce_sum(td_error)

        print(f'sum_loss is {sum_loss}, loss is {loss}')
        param = self.model.trainable_variables
        for a in self.agent_ids:
            param += self.agent_heads[a].trainable_variables

        # print(f'param {param}')

        grads = tape.gradient(sum_loss, param)

        if self.config.grad_norm_clipping:
            clipped_grads = []
            for grad in grads:
                clipped_grads.append(tf.clip_by_norm(grad, self.config.grad_norm_clipping))
            grads = clipped_grads

        grads_and_vars = list(zip(grads, param))
        self.optimizer.apply_gradients(grads_and_vars)

        return sum_loss.numpy(), sum_td_error.numpy()

    @tf.function(autograph=False)
    def update_target(self):
        for var, var_target in zip(self.model.trainable_variables, self.target_model.trainable_variables):
            var_target.assign(var)

        vars, target_vars = [], []
        for a in self.agent_ids:
            vars.extend(self.agent_heads[a].trainable_variables)
            target_vars.extend(self.target_agent_heads[a].trainable_variables)

        for var, var_target in zip(vars, target_vars):
            var_target.assign(var)

    @tf.function(autograph=False)
    def soft_update_target(self):
        for var, var_target in zip(self.model.trainable_variables, self.target_model.trainable_variables):
            var_target.assign(self.tau * var + (1.0 - self.tau) * var_target)

        vars, target_vars = [], []
        for a in self.agent_ids:
            vars.extend(self.agent_heads[a].trainable_variables)
            target_vars.extend(self.target_agent_heads[a].trainable_variables)

        for var, var_target in zip(vars, target_vars):
            var_target.assign(self.tau * var + (1.0 - self.tau) * var_target)

    def save(self, save_path):
        self.model.save_weights(f'{save_path}/value_network.h5')
        self.target_model.save_weights(f'{save_path}/target_network.h5')
        for a in self.agent_ids:
            self.agent_heads[a].save_weights(f'{save_path}/agent_{a}_head.h5')
            self.target_agent_heads[a].save_weights(f'{save_path}/target_agent_{a}_head.h5')

    def load(self, load_path):
        self.model.load_weights(f'{load_path}/value_network.h5')
        self.target_model.load_weights(f'{load_path}/target_network.h5')
        for a in self.agent_ids:
            self.agent_heads[a].load_weights(f'{load_path}/agent_{a}_head.h5')
            self.target_agent_heads[a].load_weights(f'{load_path}/target_agent_{a}_head.h5')


    def learn(self):
        episode_rewards = [0.0]
        saved_mean_reward = None
        obs = self.env.reset()
        done = False
        # Start total timer
        tstart = time.time()
        for t in range(self.config.num_timesteps):
            update_eps = tf.constant(self.exploration.value(t))

            if t % self.config.print_freq == 0:
                time_1000_step = time.time()
                nseconds = time_1000_step - tstart
                tstart = time_1000_step
                print(f'time spend to perform {t - self.config.print_freq} to {t} steps is {nseconds} ')
                print('eps update', self.exploration.value(t))

            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
            # mb_states = states
            epinfos = []
            for _ in range(self.config.n_steps):
                actions = self.choose_action(tf.constant(obs), update_eps=update_eps)
                # print(f'actions is {actions}')

                mb_obs.append(obs.copy())
                mb_actions.append(actions)
                mb_dones.append([float(done) for _ in self.agent_ids])

                obs1, rews, done, info = self.env.step(actions)

                if self.config.same_reward_for_agents:
                    rews = [np.max(rews) for _ in range(len(rews))]  # for cooperative purpose same reward for every one

                mb_rewards.append(rews)
                obs = obs1
                maybeepinfo = info.get('episode')
                if maybeepinfo : epinfos.append(maybeepinfo)

                episode_rewards[-1] += np.max(rews)
                if done:
                    episode_rewards.append(0.0)
                    obs = self.env.reset()

            mb_dones.append([float(done) for _ in self.agent_ids])

            # swap axes to have lists in shape of (num_agents, num_steps, ...)
            mb_obs = np.asarray(mb_obs, dtype=obs[0].dtype).swapaxes(0, 1)
            mb_actions = np.asarray(mb_actions, dtype=actions[0].dtype).swapaxes(0, 1)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(0, 1)
            mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(0, 1)
            mb_masks = mb_dones[:, -1]
            mb_dones = mb_dones[:, 1:]

            # print(f'before discount mb_rewards is {mb_rewards}')

            if self.config.gamma > 0.0:
                # Discount/bootstrap off value fn
                last_values = self.value(tf.constant(obs1))
                # print(f'last_values {last_values}')

                for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                    rewards = rewards.tolist()
                    dones = dones.tolist()
                    if dones[-1] == 0:
                        rewards = discount_with_dones(rewards + [value], dones + [0], self.config.gamma)[:-1]
                    else:
                        rewards = discount_with_dones(rewards, dones, self.config.gamma)

                    mb_rewards[n] = rewards

            # print(f'after discount mb_rewards is {mb_rewards}')

            if self.config.replay_buffer is not None:
                self.replay_memory.add((mb_obs, mb_actions, mb_rewards, obs1, mb_masks))

            if t > self.config.learning_starts and t % self.config.train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if self.config.prioritized_replay:
                    experience = self.replay_memory.sample(self.config.batch_size, beta=self.beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = self.replay_memory.sample(self.config.batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None

                obses_t = tf.constant(obses_t)
                actions = tf.constant(actions)
                rewards = tf.constant(rewards)
                weights = tf.constant(weights)

                loss, td_errors = self.train(obses_t, actions, rewards, weights)

            if t > self.config.learning_starts and t % self.config.target_network_update_freq == 0:
                # Update target network periodically.
                self.soft_update_target()

            if t % self.config.playing_test == 0 and t != 0:
                self.play_test_games()

            mean_100ep_reward = np.mean(episode_rewards[-101:-1])
            num_episodes = len(episode_rewards)
            if done and self.config.print_freq is not None and len(episode_rewards) % self.config.print_freq == 0:
                print(f'last 100 episode mean reward {mean_100ep_reward} in {num_episodes} playing')
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * self.exploration.value(t)))
                logger.dump_tabular()

    def play_test_games(self):
        num_tests = self.config.num_tests

        test_env = init_env(self.config, mode='test')

        test_rewards = np.zeros(num_tests)
        for i in range(num_tests):
            test_done = False
            test_obs_all = test_env.reset()
            # print(np.asarray(test_obs_all).shape)
            while not test_done:
                test_obs_all = tf.constant(test_obs_all)
                test_action_list = self.choose_action(test_obs_all, stochastic=False)
                test_new_obs_list, test_rew_list, test_done, _ = test_env.step(test_action_list)
                test_obs_all = test_new_obs_list

                if test_done:
                    print(f'test_reward_dict for test {i} is {test_rew_list}')
                    test_rewards[i] = np.mean(test_rew_list)

        print(f'mean reward of {num_tests} tests is {np.mean(test_rewards)}')
        test_env.close()



def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1. - done)  # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]



