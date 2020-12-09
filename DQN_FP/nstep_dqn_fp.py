import os.path as osp
import numpy as np
import tensorflow as tf
import time

from common import logger
from common.schedules import LinearSchedule
from common.utils import set_global_seeds

from DQN_FP.tf2_models import build_q_func, impala_fp
from DQN_FP.utils import play_test_games
from DQN_FP.agents import MAgent
from DQN_FP.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


def learn(env,
          config,
          seed=None,
          lr=0.00008,
          total_timesteps=100000,
          buffer_size=8,
          exploration_fraction=0.1,
          exploration_final_eps=0.01,
          train_freq=1,
          batch_size=8,
          print_freq=100,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=7,
          gamma=0.993,
          target_network_update_freq=1000,
          hiddens=[256],
          dueling=True,
          layer_norm=True,
          double_q=True,
          grad_norm_clipping = True,
          play_test = 10,
          nsteps = 4,
          tau = 0.01,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          load_path=None,
          ):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
        (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
        will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
    load_path: str
        path to load the model from. (default: None)
    **network_kwargs
        additional keyword arguments to pass to the network builder.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model
    set_global_seeds(seed)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    agent_ids = env.agent_ids()
    num_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    print(f'agent_ids {agent_ids}, num_actions {num_actions}, obs_shape {obs_shape}')

    network = impala_fp(obs_shape, len(agent_ids), 'nstep_dqn_cnn', num_actions, num_extra_data=2)
    q_func = build_q_func(network, hiddens=hiddens, dueling=dueling, layer_norm=layer_norm)


    dqn_agent = MAgent(q_func, obs_shape, agent_ids, lr, tau, double_q, num_actions, gamma,
                          grad_norm_clipping, param_noise)

    # dqn_agent.update_target()
    dqn_agent.soft_update_target()

    if load_path is not None:
        dqn_agent.load_model(load_path)


    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True
    done = False

    # Start total timer
    tstart = time.time()
    for t in range(total_timesteps):
        if callback is not None:
            if callback(locals(), globals()):
                break
        kwargs = {}
        if not param_noise:
            update_eps = tf.constant(exploration.value(t))
            update_param_noise_threshold = 0.
        else:
            update_eps = tf.constant(0.)
            # Compute the threshold such that the KL divergence between perturbed and non-perturbed
            # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
            # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
            # for detailed explanation.
            update_param_noise_threshold = -np.log(
                1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
            kwargs['reset'] = reset
            kwargs['update_param_noise_threshold'] = update_param_noise_threshold
            kwargs['update_param_noise_scale'] = True

        if t % print_freq == 0:
            time_1000_step = time.time()
            nseconds = time_1000_step - tstart
            tstart = time_1000_step
            print(f'time spend to perform {t-print_freq} to {t} steps is {nseconds} ')
            print('eps update', exploration.value(t))

        mb_obs, mb_rewards, mb_actions, mb_fps, mb_dones = [], [], [], [], []
        # mb_states = states
        epinfos = []
        for _ in range(nsteps):
            # Given observations, take action and value (V(s))
            actions_list, fps_ = dqn_agent.choose_action(tf.constant(obs), update_eps=update_eps, **kwargs)
            fps = []
            # print(f'fps_.shape is {np.asarray(fps_).shape}')
            for a in agent_ids:
                fps.append(fps_[:a] + fps_[a + 1:])  # keeps fingerprints of others for each agent

            # print(fps)
            # print(f'actions_list is {actions_list}')
            # print(f'values_list is {values_list}')

            # Append the experiences
            mb_obs.append(obs.copy())
            mb_actions.append(actions_list)
            mb_fps.append(fps)
            mb_dones.append([float(done) for _ in agent_ids])

            # Take actions in env and look the results
            obs1, rews, done, info = env.step(actions_list)
            rews = [np.max(rews) for _ in range(len(rews))]  # for cooperative purpose same reward for every one
            mb_rewards.append(rews)
            obs = obs1
            # print(rewards, done, info)
            maybeepinfo = info.get('episode')
            if maybeepinfo: epinfos.append(maybeepinfo)

            episode_rewards[-1] += np.max(rews)
            if done:
                episode_rewards.append(0.0)
                obs = env.reset()
                reset = True

        mb_dones.append([float(done) for _ in agent_ids])

        # print(f'mb_actions is {mb_actions}')
        # print(f'mb_rewards is {mb_rewards}')
        # print(f'mb_fps is {mb_fps}')
        # print(f'mb_dones is {mb_dones}')

        mb_obs = np.asarray(mb_obs, dtype=obs[0].dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=actions_list[0].dtype)
        mb_fps = np.asarray(mb_fps, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_masks = mb_dones[:-1]
        mb_dones = mb_dones[1:]

        # print(f'mb_actions is {mb_actions}')
        # print(f'mb_rewards is {mb_rewards}')
        # print(f'mb_fps is {mb_fps}')
        # print(f'mb_dones is {mb_dones}')
        # print(f'mb_masks is {mb_masks}')

        if gamma > 0.0:
            # Discount/bootstrap off value fn
            last_values = dqn_agent.value(tf.constant(obs1))
            # print(f'last_values is {last_values}')
            if mb_dones[-1][0] == 0:
                mb_rewards = discount_with_dones(np.concatenate((mb_rewards, [last_values])),
                                                 np.concatenate((mb_dones, [[float(False) for _ in agent_ids]]))
                                                 , gamma)[:-1]
            else:
                mb_rewards = discount_with_dones(mb_rewards, mb_dones, gamma)

        # print(f'after discount mb_rewards is {mb_rewards}')

        if replay_buffer is not None:
            replay_buffer.add(mb_obs, mb_actions, mb_rewards, mb_masks,
                              mb_fps, [exploration.value(t), t])

        if t > learning_starts and t % train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if prioritized_replay:
                experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                (obses_t, actions, rewards, dones, fps, extra_datas, weights, batch_idxes) = experience
            else:
                obses_t, actions, rewards, dones, fps, extra_datas = replay_buffer.sample(batch_size)
                weights, batch_idxes = np.ones_like(rewards), None

            obses_t, weights = tf.constant(obses_t), tf.constant(weights)
            actions, rewards = tf.constant(actions), tf.constant(rewards, dtype=tf.float32)
            fps, extra_datas = tf.constant(fps), tf.constant(extra_datas)

            s = obses_t.shape
            obses_t = tf.reshape(obses_t, (s[0] * s[1], *s[2:]))

            s = actions.shape
            actions = tf.reshape(actions, (s[0] * s[1], *s[2:]))

            s = rewards.shape
            rewards = tf.reshape(rewards, (s[0] * s[1], *s[2:]))

            s = weights.shape
            weights = tf.reshape(weights, (s[0] * s[1], *s[2:]))

            s = fps.shape
            # print(f'fps.shape is {s}')
            fps = tf.reshape(fps, (s[0] * s[1], *s[2:]))
            # print(f'fps.shape is {fps.shape}')
            s = extra_datas.shape
            # print(f'extra_datas.shape is {extra_datas.shape}')
            # extra_datas = tf.reshape(extra_datas, (s[0] * s[1], *s[2:]))
            # print(f'extra_datas.shape is {extra_datas.shape}')

            loss, td_errors = dqn_agent.nstep_train(obses_t, actions, rewards, dones, weights, fps, extra_datas)

        if t > learning_starts and t % target_network_update_freq == 0:
            # Update target network periodically.
            dqn_agent.soft_update_target()


        if t % play_test == 0 and t != 0:
            play_test_games(dqn_agent, config)

        mean_100ep_reward = np.mean(episode_rewards[-101:-1])
        num_episodes = len(episode_rewards)
        if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            print(f'last 100 episode mean reward {mean_100ep_reward} in {num_episodes} playing')
            logger.record_tabular("steps", t)
            logger.record_tabular("episodes", num_episodes)
            logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
            logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
            logger.dump_tabular()


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    # print(f'rewards[::-1] is {rewards[::-1]}')
    # print(f'dones[::-1] is {dones[::-1]}')
    for reward, done in zip(rewards[::-1], dones[::-1]):
        # print(f'reward is {reward}')
        # print(f'done is {done}')
        r = reward + gamma*r*(1.-done)  # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]

#
# # tf.config.run_functions_eagerly(True)
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# import matplotlib.pyplot as plt
# import numpy as np
#
# num_agents = 2
# render = False
# stacked = True
# env_name = 'academy_empty_goal'  #'academy_3_vs_1_with_keeper'
# channel_dim = (51, 40)
# representationType = 'extracted'  # 'extracted': minimaps, 'pixels': raw pixels, 'simple115': vector of 115
# rewards = 'checkpoints,scoring'  # checkpoints,
# env = create_ma_env(num_agents, render, stacked, env_name, representationType, channel_dim, rewards)
#
# # obs = env.reset()
# # for agent_name in agent_names:
# #     obs_a = obs[agent_name]
# #     plt.imshow(obs_a[:, :, 3])
# #     plt.show()
#
#
# # action_dict = env.sample()
# # print('action_dict ', action_dict)
# # obs, rew, done, _ = env.step_(action_dict)
# #
# # print('obs, rew, done', obs, rew, done)
# SEED = 0
# print(f'SEED is {SEED}')
# learn(env, SEED)
