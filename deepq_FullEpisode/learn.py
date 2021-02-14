
import time
import numpy as np
import tensorflow as tf

from common import logger
from common.schedules import LinearSchedule

from deepq_FullEpisode.agent import Agent
from deepq_FullEpisode.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from deepq_FullEpisode.network import Network
from deepq_FullEpisode.utils import discount_with_dones, init_env


# Learning
class Learn:
    def __init__(self, config, env):
        super().__init__()
        self.config = config
        self.env = env
        self.agent_ids = self.get_agent_ids()

        self.replay_memory, self.beta_schedule = self.init_replay_memory()
        self.optimizer = tf.keras.optimizers.Adam(self.config.lr)
        # Create the schedule for exploration starting from 1.
        self.exploration = LinearSchedule(schedule_timesteps=int(config.exploration_fraction * config.num_timesteps),
                                          initial_p=1.0, final_p=config.exploration_final_eps)
        self.eps = tf.Variable(0.0)

        self.models, self.target_models = self._init_networks()

        self.agents = [Agent(config, self.models[agent_id],
                             self.target_models[agent_id], agent_id) for agent_id in self.agent_ids]

    def _init_networks(self):
        network = Network(self.config, self.agent_ids)
        # base_model = network.init_base_model()
        # target_base_model = network.init_base_model()
        return network.build_models(), network.build_models()

    def get_agent_ids(self):
        return [agent_id for agent_id in range(self.config.num_agents)]

    def init_replay_memory(self):
        """
        :return: replay_buffer, beta_schedule
        """
        if self.config.prioritized_replay:
            replay_buffer = PrioritizedReplayBuffer(self.config.buffer_size, alpha=self.config.prioritized_replay_alpha,
                                                    n_steps=self.config.n_steps)
            if self.config.prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = self.config.num_timesteps
            beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                           initial_p=self.config.prioritized_replay_beta0, final_p=1.0)
        else:
            replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.n_steps)
            beta_schedule = None
        return replay_buffer, beta_schedule

    @tf.function
    def get_actions(self, obs, stochastic=True, update_eps=-1):
        """
        :param obs: observation for all agents
        :param stochastic: True for Train phase and False for test phase
        :param update_eps: epsilon update for eps-greedy
        :return: actions, q_values of all agents as fps
        """
        deterministic_actions = []
        fps = []
        for agent_id in self.agent_ids:
            deterministic_action, fp = self.agents[agent_id].greedy_action(obs[agent_id])
            deterministic_actions.append(deterministic_action)
            fps.append(fp)
        # print(f' deterministic_actions {deterministic_actions}')
        # print(f' fps {fps}')

        random_actions = tf.random.uniform(tf.stack([self.config.num_agents]), minval=0,
                                           maxval=self.config.num_actions, dtype=tf.int64)
        # print(f' random_actions {random_actions}')
        chose_random = tf.random.uniform(tf.stack([self.config.num_agents]), minval=0,
                                         maxval=1, dtype=tf.float32) < self.eps
        # print(f' chose_random {chose_random}')

        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)
        # print(f' stochastic_actions.numpy() {stochastic_actions.numpy()}')

        if stochastic:
            actions = stochastic_actions.numpy()
        else:
            actions = deterministic_actions

        if update_eps >= 0:
            self.eps.assign(update_eps)

        # print(f' actions {actions}')
        return actions, fps

    @tf.function
    def get_max_values(self, obs):
        """
        :param obs: list observations one for each agent
        :return: best values based on Q-Learning formula maxQ(s',a')
        """
        best_q_vals = []
        for agent_id in self.agent_ids:
            best_q_val = self.agents[agent_id].max_value(obs[agent_id])
            best_q_vals.append(best_q_val)
        # print(f' best_q_vals.numpy() {best_q_vals.numpy()}')
        return best_q_vals

    @tf.function
    def compute_loss(self, obses_t, actions, rewards, dones, weights, fps=None):
        """
        :param obses_t: list observations one for each agent
        :param actions:
        :param rewards:
        :param dones:
        :param weights:
        :param fps:
        :return: loss and td errors tensor list one for each agent
        """
        losses = []
        td_errors = np.zeros(self.config.batch_size*self.config.n_steps)
        for agent_id in self.agent_ids:
            loss, td_error = self.agents[agent_id].compute_loss(obses_t[agent_id], actions[agent_id],
                                                                rewards[agent_id], dones[agent_id],
                                                                weights[agent_id], fps=None)
            losses.append(loss)
            td_errors += td_error

        return losses, td_errors

    @tf.function()
    def train(self, obses_t, actions, rewards, dones, weights, fps=None):
        with tf.GradientTape() as tape:
            losses, td_errors = self.compute_loss(obses_t, actions, rewards, dones, weights, fps)
            loss = tf.reduce_sum(losses)

        params = tape.watched_variables()
        # print(f' param {params}')
        grads = tape.gradient(loss, params)

        if self.config.grad_norm_clipping:
            clipped_grads = []
            for grad in grads:
                clipped_grads.append(tf.clip_by_norm(grad, self.config.grad_norm_clipping))
            grads = clipped_grads

        self.optimizer.apply_gradients(list(zip(grads, params)))

        return loss, td_errors.numpy()

    def create_fingerprints(self, fps, t):
        # TODO
        fps = []
        if self.config.num_agents > 1:
            for agent_id in self.agent_ids:
                fp = fps[:agent_id]
                fp.extend(fps[agent_id + 1:])
                fp_a = np.concatenate((fp, [[self.exploration.value(t) * 100, t]]), axis=None)
                fps.append(fp_a)
        return fps

    def learn(self):
        episode_rewards = [0.0]
        obs = self.env.reset()
        print(obs.shape)

        done = False
        tstart = time.time()
        episodes_trained = [0, False]  # [episode_number, Done flag]
        t = 0
        for ep in range(self.config.num_episodes):
            episode_length = 0
            # if t == 102:
            #     break
            update_eps = tf.constant(self.exploration.value(t))

            mb_obs, mb_rewards, mb_actions, mb_obs1, mb_dones = [], [], [], [], []
            while True:
            # for n_step in range(self.config.n_steps):
                t += 1
                episode_length += 1
                # print(f't is {t} -- n_steps is {n_step}')
                actions, _ = self.get_actions(tf.constant(obs), update_eps=update_eps)
                if self.config.num_agents == 1:
                    obs1, rews, done, _ = self.env.step(actions[0])
                else:
                    obs1, rews, done, _ = self.env.step(actions)
                    # TODO fingerprint computation

                mb_obs.append(obs.copy())
                mb_actions.append(actions)
                mb_dones.append([float(done) for _ in self.agent_ids])

                # print(f'rewards is {rews}')

                if self.config.same_reward_for_agents:
                    rews = [np.max(rews) for _ in range(len(rews))]  # for cooperative purpose same reward for every one

                mb_obs1.append(obs1.copy())
                mb_rewards.append(rews)

                obs = obs1
                episode_rewards[-1] += np.max(rews)
                if done or episode_length > self.config.max_episodes_length:
                    episodes_trained[0] = episodes_trained[0] + 1
                    episodes_trained[1] = True
                    episode_rewards.append(0.0)
                    obs = self.env.reset()
                    break  # to break while as episode is finished here

            # print(f' mb_obs.shape is {np.array(mb_obs).shape}')
            for extra_step in range(self.config.n_steps - len(mb_actions)):
                # print('extra_info as 0 s added')
                mb_obs.append(obs * 0.)
                mb_actions.append(actions * 0.)
                mb_rewards.append(np.array(rews) * 0.)
                # mb_fps.append(0, mb_fps[-1] * 0.)
                mb_dones.append([float(0.) for _ in self.agent_ids])

            mb_dones.append([float(done) for _ in self.agent_ids])
            # swap axes to have lists in shape of (num_agents, num_steps, ...)
            # print(f' mb_obs.shape is {np.array(mb_obs).shape}')
            mb_obs = np.asarray(mb_obs, dtype=obs[0].dtype).swapaxes(0, 1)
            # print(f' mb_obs.shape is {np.array(mb_obs).shape}')
            mb_actions = np.asarray(mb_actions, dtype=actions[0].dtype).swapaxes(0, 1)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(0, 1)
            mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(0, 1)
            mb_masks = mb_dones[:, :-1]
            mb_dones = mb_dones[:, 1:]


            # print(f' before discount mb_rewards is {mb_rewards}')

            if self.config.gamma > 0.0:
                # print(f' last_values {last_values}')
                for agent_id, (rewards, dones) in enumerate(zip(mb_rewards, mb_dones)):
                    value = self.agents[agent_id].max_value(tf.constant(obs1[agent_id]))
                    rewards = rewards.tolist()
                    dones = dones.tolist()
                    if dones[-1] == 0:
                        rewards = discount_with_dones(rewards + [value], dones + [0], self.config.gamma)[:-1]
                    else:
                        rewards = discount_with_dones(rewards, dones, self.config.gamma)

                    mb_rewards[agent_id] = rewards

            # print(f' after discount mb_rewards is {mb_rewards}')

            if self.config.replay_buffer is not None:
                self.replay_memory.add_episode(mb_obs, mb_actions, mb_rewards, mb_masks)

            if ep > self.config.learning_starts:
                if self.config.prioritized_replay:
                    experience = self.replay_memory.sample(self.config.batch_size, beta=self.beta_schedule.value(t))
                    (obses_t, actions, rewards, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, dones = self.replay_memory.sample(self.config.batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None

                # print(f'obses_t.shape {obses_t.shape}')
                #  shape format is (batch_size, agent_num, n_steps, ...)
                obses_t = obses_t.swapaxes(0, 1)
                actions = actions.swapaxes(0, 1)
                # print(f'rewards.shape {rewards.shape}')
                rewards = rewards.swapaxes(0, 1)
                # print(f'rewards.shape {rewards.shape}')
                # obses_tp1 = obses_tp1.swapaxes(0, 1)
                dones = dones.swapaxes(0, 1)
                weights = np.expand_dims(weights, 2)
                # print(f'weights.shape {weights.shape}')
                _wt = np.tile(weights, (self.config.num_agents, 1, self.config.n_steps))
                # print(f'_wt.shape {_wt.shape}')
                # print(f'weights.shape {weights.shape}')
                # weights = weights.swapaxes(0, 1)  # weights shape is (1, batch_size, n_steps)
                # print(f'weights.shape {weights.shape}')
                #  shape format is (agent_num, batch_size, n_steps, ...)

                obses_t = tf.constant(obses_t)
                actions = tf.constant(actions)
                rewards = tf.constant(rewards)
                dones = tf.constant(dones)
                _wt = tf.constant(_wt)

                if 'rnn' not in self.config.network:
                    shape = obses_t.shape
                    obses_t = np.reshape(obses_t, (shape[0], shape[1] * shape[2], *shape[3:]))
                    shape = actions.shape
                    actions = np.reshape(actions, (shape[0], shape[1] * shape[2], *shape[3:]))
                    shape = rewards.shape
                    rewards = np.reshape(rewards, (shape[0], shape[1] * shape[2], *shape[3:]))
                    shape = dones.shape
                    dones = np.reshape(dones, (shape[0], shape[1] * shape[2], *shape[3:]))
                    shape = _wt.shape
                    _wt = np.reshape(_wt, (shape[0], shape[1] * shape[2], *shape[3:]))

                    # print(f'obses_t.shape {obses_t.shape}')
                    #  shape format is (agent_num, batch_size * n_steps, ...)

                # print(f' obses_t.shape {obses_t.shape}')
                # print(f' actions.shape {actions.shape}')
                # print(f' rewards.shape {rewards.shape}')
                # print(f' dones.shape {dones.shape}')
                # print(f' _wt.shape {_wt.shape}')

                loss, td_errors = self.train(obses_t, actions, rewards, dones, _wt)

                # print(f' td_errors.shape {td_errors.shape}')
                td_errors = td_errors.reshape((self.config.batch_size, -1))
                # print(f' td_errors.shape {td_errors.shape}')
                td_errors = np.sum(td_errors, 1)
                # print(f' td_errors.shape {td_errors.shape}')

                # print(f'td_errors.shape = {np.array(td_errors).shape} , batch_idxes.shape = {np.array(batch_idxes).shape}')
                if self.config.prioritized_replay:
                    new_priorities = np.abs(td_errors) + self.config.prioritized_replay_eps
                    self.replay_memory.update_priorities(batch_idxes, new_priorities)

                if ep % (self.config.print_freq) == 0:
                    print(f't = {t} , loss = {loss}')

            if ep > self.config.learning_starts and ep % self.config.target_network_update_freq == 0:
                # Update target network periodically.
                for agent_id in self.agent_ids:
                    self.agents[agent_id].soft_update_target()

            if ep % self.config.playing_test == 0 and ep != 0:
                # self.network.save(self.config.save_path)
                self.play_test_games()

            mean_100ep_reward = np.mean(episode_rewards[-101:-1])
            num_episodes = len(episode_rewards)

            if ep % (self.config.print_freq*10) == 0:
                time_1000_step = time.time()
                nseconds = time_1000_step - tstart
                tstart = time_1000_step
                print(f'eps {self.exploration.value(t)} -- time {t - self.config.print_freq*10} to {t} steps: {nseconds}')

            # if done and self.config.print_freq is not None and len(episode_rewards) % self.config.print_freq == 0:
            if episodes_trained[1] and episodes_trained[0] % self.config.print_freq == 0:
                episodes_trained[1] = False
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 past episode reward", mean_100ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * self.exploration.value(t)))
                logger.dump_tabular()

    def play_test_games(self):
        num_tests = self.config.num_tests
        test_env = init_env(self.config, mode='test')

        test_rewards = np.zeros(num_tests)
        for i in range(num_tests):
            done = False
            obs = test_env.reset()
            iter = 0
            while True:
                iter += 1
                actions, _ = self.get_actions(tf.constant(obs), stochastic=False)
                # print(f'actions[0] {actions[0]}, test_done {done}, {iter}')
                if self.config.num_agents == 1:
                    obs1, rews, done, _ = test_env.step(actions[0])
                else:
                    obs1, rews, done, _ = test_env.step(actions)
                    # ToDo fingerprint computation

                obs = obs1

                if done or iter >= self.config.max_episodes_length:
                    # print(f'test {i} rewards is {rews}')
                    test_rewards[i] = np.mean(rews)
                    break

        print(f'test_rewards: {test_rewards} \n mean reward of {num_tests} tests: {np.mean(test_rewards)}')
        test_env.close()

