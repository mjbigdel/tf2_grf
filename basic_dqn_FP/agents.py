
import tensorflow as tf
import numpy as np
import time


from common.schedules import LinearSchedule
from common import logger
from common.tf2_utils import huber_loss
from common.utils import init_env

from basic_dqn_FP.utils import init_replay_memory
from basic_dqn_FP.network import Network


class Agent(tf.Module):
    def __init__(self, config, env):
        self.config = config
        self.agent_ids = [a for a in range(config.num_agents)]
        self.env = env
        self.optimizer = tf.keras.optimizers.Adam(self.config.lr)
        self.replay_memory, self.beta_schedule = init_replay_memory(config)

        # Create the schedule for exploration starting from 1.
        self.exploration = LinearSchedule(schedule_timesteps=int(config.exploration_fraction * config.num_timesteps),
                                          initial_p=1.0, final_p=config.exploration_final_eps)

        self.loss = self.nstep_loss
        self.eps = tf.Variable(0.0)

        # init model
        self.network = Network(config)

    @tf.function
    def choose_action(self, obs, stochastic=True, update_eps=-1):
        """

        :param obs: list observations one for each agent
        :param stochastic: True for Train phase and False for test phase
        :param update_eps: epsilon update for eps-greedy
        :return: actions: list of actions chosen by agents based on observation one for each agent
        """

        # actions = []
        # fps = []
        deterministic_actions, fps = self.network.step(obs)
        # print(f'deterministic_actions {deterministic_actions}')

        batch_size = len(self.agent_ids)
        random_actions = tf.random.uniform(tf.stack([batch_size]), minval=0,
                                           maxval=self.config.num_actions, dtype=tf.int64)
        # print(f'random_actions {random_actions}')
        chose_random = tf.random.uniform(tf.stack([batch_size]), minval=0,
                                         maxval=1, dtype=tf.float32) < self.eps
        # print(f'chose_random {chose_random}')

        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)
        # print(f'stochastic_actions.numpy() {stochastic_actions.numpy()}')

        if stochastic:
            actions = stochastic_actions.numpy()
        else:
            actions = deterministic_actions

        if update_eps >= 0:
            self.eps.assign(update_eps)

        # print(f'fps.shape {np.array(fps).shape}')
        return actions, fps

    @tf.function()
    def nstep_loss(self, obses_t_a, actions_a, rewards_a, dones_a, weights_a, fps_a, agent_id):
        # print(f'obses_t_a.shape {obses_t_a.shape}')
        q_t = self.network.value(obses_t_a, fps_a, agent_id)

        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(actions_a, self.config.num_actions, dtype=tf.float32), 1)
        # print(f'q_t_selected.shape is {q_t_selected.shape}')

        td_error = q_t_selected - tf.stop_gradient(rewards_a)

        errors = huber_loss(td_error)
        weighted_loss = tf.reduce_mean(weights_a * errors)

        return weighted_loss, td_error

    @tf.function()
    def train(self, obses_t, actions, rewards, dones, weights, fps):
        td_errors = []
        loss = []
        with tf.GradientTape() as tape:
            for a in self.agent_ids:
                if self.config.network == 'tdcnn_rnn':
                    loss_a, td_error = self.loss(obses_t[a], actions[a, :, -1], rewards[a, :, -1],
                                                 dones[a, :, -1], weights[a, :, -1], fps[a], a)
                else:
                    loss_a, td_error = self.loss(obses_t[a], actions[a], rewards[a], dones[a], weights[a], fps[a], a)

                loss.append(loss_a)
                td_errors.append(td_error)

            sum_loss = tf.reduce_sum(loss)
            sum_td_error = tf.reduce_sum(td_errors)

        # print(f'sum_loss is {sum_loss}, loss is {loss}')
        param = self.network.model.trainable_variables
        for a in self.agent_ids:
            param += self.network.agent_heads[a].trainable_variables

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


    def learn(self):
        self.network.soft_update_target()
        episode_rewards = [0.0]
        obs = self.env.reset()
        done = False
        tstart = time.time()
        episodes_trained = [0, False]  # [episode_number, Done flag]
        for t in range(self.config.num_timesteps):
            update_eps = tf.constant(self.exploration.value(t))
            if t % (self.config.print_freq) == 0:
                time_1000_step = time.time()
                nseconds = time_1000_step - tstart
                tstart = time_1000_step
                print(f'eps {self.exploration.value(t)} -- time {t - self.config.print_freq} to {t} steps: {nseconds}')

            mb_obs, mb_rewards, mb_actions, mb_fps, mb_dones = [], [], [], [], []
            # mb_states = states
            epinfos = []
            for nstep in range(self.config.n_steps):
                actions, fps_ = self.choose_action(tf.constant(obs), update_eps=update_eps)
                fps = []
                if self.config.num_agents > 1:
                    for a in self.agent_ids:
                        fp = fps_[:a]
                        fp.extend(fps_[a + 1:])
                        fp_a = np.concatenate((fp, [[self.exploration.value(t)*100, t]]), axis=None)
                        fps.append(fp_a)

                # print(f'fps.shape {np.array(fps).shape}')
                mb_obs.append(obs.copy())
                mb_actions.append(actions)
                mb_fps.append(fps)
                mb_dones.append([float(done) for _ in self.agent_ids])

                obs1, rews, done, info = self.env.step(actions.tolist())

                if self.config.same_reward_for_agents:
                    rews = [np.max(rews) for _ in range(len(rews))]  # for cooperative purpose same reward for every one

                mb_rewards.append(rews)
                obs = obs1
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)

                episode_rewards[-1] += np.max(rews)
                if done:
                    episodes_trained[0] = episodes_trained[0] + 1
                    episodes_trained[1] = True
                    episode_rewards.append(0.0)
                    obs = self.env.reset()

            mb_dones.append([float(done) for _ in self.agent_ids])

            # swap axes to have lists in shape of (num_agents, num_steps, ...)
            mb_obs = np.asarray(mb_obs, dtype=obs[0].dtype).swapaxes(0, 1)
            mb_actions = np.asarray(mb_actions, dtype=actions[0].dtype).swapaxes(0, 1)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(0, 1)
            mb_fps = np.asarray(mb_fps, dtype=np.float32).swapaxes(0, 1)
            mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(0, 1)
            mb_masks = mb_dones[:, :-1]
            mb_dones = mb_dones[:, 1:]

            # print(f'before discount mb_rewards is {mb_rewards}')

            if self.config.gamma > 0.0:
                # Discount/bootstrap off value fn
                last_values = self.network.last_value(tf.constant(obs1))
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
                self.replay_memory.add((mb_obs, mb_actions, mb_rewards, obs1, mb_masks, mb_fps))

            if t > self.config.learning_starts and t % self.config.train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if self.config.prioritized_replay:
                    experience = self.replay_memory.sample(self.config.batch_size, beta=self.beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, fps, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones, fps = self.replay_memory.sample(self.config.batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None

                #  shape format is (batch_size, agent_num, n_steps, ...)
                obses_t = obses_t.swapaxes(0, 1)
                actions = actions.swapaxes(0, 1)
                rewards = rewards.swapaxes(0, 1)
                obses_tp1 = obses_tp1.swapaxes(0, 1)
                dones = dones.swapaxes(0, 1)
                fps = fps.swapaxes(0, 1)
                weights = weights.swapaxes(0, 1)

                if self.config.network == 'cnn':
                    shape = obses_t.shape
                    obses_t = np.reshape(obses_t, (shape[0], shape[1]*shape[2], *shape[3:]))
                    shape = actions.shape
                    actions = np.reshape(actions, (shape[0], shape[1]*shape[2], *shape[3:]))
                    shape = rewards.shape
                    rewards = np.reshape(rewards, (shape[0], shape[1]*shape[2], *shape[3:]))
                    shape = dones.shape
                    dones = np.reshape(dones, (shape[0], shape[1]*shape[2], *shape[3:]))
                    shape = weights.shape
                    weights = np.reshape(weights, (shape[0], shape[1] * shape[2], *shape[3:]))
                    shape = fps.shape
                    fps = np.reshape(fps, (shape[0], shape[1] * shape[2], *shape[3:]))

                #  shape format is (agent_num, batch_size, n_steps, ...)
                obses_t = tf.constant(obses_t)
                actions = tf.constant(actions)
                rewards = tf.constant(rewards)
                dones = tf.constant(dones)
                weights = tf.constant(weights)
                fps = tf.constant(fps)

                # print(f'obses_t.shape {obses_t.shape}')
                # print(f'actions.shape {actions.shape}')
                # print(f'rewards.shape {rewards.shape}')
                # print(f'dones.shape {dones.shape}')
                # print(f'weights.shape {weights.shape}')
                # print(f'fps.shape {fps.shape}')

                loss, td_errors = self.train(obses_t, actions, rewards, dones, weights, fps)
            
                if t % (self.config.train_freq*50) == 0:
                    print(f't = {t} , loss = {loss}')
                

            if t > self.config.learning_starts and t % self.config.target_network_update_freq == 0:
                # Update target network periodically.
                self.network.soft_update_target()

            if t % self.config.playing_test == 0 and t != 0:
                # self.network.save(self.config.save_path)
                self.play_test_games()

            mean_100ep_reward = np.mean(episode_rewards[-101:-1])
            num_episodes = len(episode_rewards)
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
            test_done = False
            test_obs_all = test_env.reset()
            while not test_done:
                test_obs_all = tf.constant(test_obs_all)
                test_action_list, _ = self.choose_action(test_obs_all, stochastic=False)
                test_new_obs_list, test_rew_list, test_done, _ = test_env.step(test_action_list)
                test_obs_all = test_new_obs_list

                if test_done:
                    test_rewards[i] = np.mean(test_rew_list)

        print(f'test_rewards: {test_rewards} \n mean reward of {num_tests} tests: {np.mean(test_rewards)}')
        test_env.close()


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1. - done)  # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]

# rewards = [0., 0., 0., 0., 0., 0.5, 0., 0.]
# dones = [0, 0, 0, 0, 0, 0, 0, 0]
# gamma = .5
# dis_rews = discount_with_dones(rewards,dones,gamma)
# print(dis_rews)


