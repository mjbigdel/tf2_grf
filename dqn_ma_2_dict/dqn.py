import os.path as osp
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# from common.logger import *
from dqn_ma_2_dict.dqn_agent import MAgent
from common.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from common.schedules import LinearSchedule
from common.utils import set_global_seeds, create_ma_env


def play_test_games(dqn_agent):
    num_agents = 4
    render = True
    stacked = False
    env_name = 'academy_3_vs_1_with_keeper'
    channel_dim = (42, 42)
    representationType = 'extracted'  # 'extracted': minimaps, 'pixels': raw pixels, 'simple115': vector of 115
    test_env = create_ma_env(num_agents, render, stacked, env_name, representationType, channel_dim)
    num_tests = 3
    test_rewards = np.zeros(num_tests)
    for i in range(num_tests):
        test_done = False
        test_obs_dict = test_env.reset()
        while not test_done:
            test_action_dict = dqn_agent.choose_greedy_action(test_obs_dict)
            test_new_obs_dict, test_rew_dict, test_done_dict, _ = test_env.step(test_action_dict)
            test_obs_dict = test_new_obs_dict

            test_done = test_done_dict['__all__']
            if test_done:
                test_rewards[i] = test_rew_dict['agent_0']

    print(f'mean reward of {num_tests} tests is {np.mean(test_rewards)}')
    test_env.close()

def learn(env,
          seed=None,
          lr=5e-4,
          total_timesteps=500000,
          buffer_size=10000,
          exploration_fraction=0.3,
          exploration_final_eps=0.02,
          train_freq= 10,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=10000,
          gamma=0.99,
          target_network_update_freq=1000,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          load_path=None,
          **network_kwargs
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
    double_q = False
    grad_norm_clipping = None
    shared_weights = True
    tf.config.run_functions_eagerly(True)
    play_test = 2000

    # Create the replay buffer
    replay_memory = True
    if replay_memory:
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
    else:
        replay_buffer = None
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    agent_names = env.agent_names()
    print(f'agent_names {agent_names}')
    num_actions = env.action_space.n
    print(f'num_actions {num_actions}')

    dqn_agent = MAgent(env, agent_names, lr, replay_buffer, shared_weights, double_q, num_actions,
                           gamma, grad_norm_clipping, param_noise)


    if load_path is not None:
        load_path = osp.expanduser(load_path)
        ckpt = tf.train.Checkpoint(model=dqn_agent.q_network)
        manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=None)
        ckpt.restore(manager.latest_checkpoint)
        print("Restoring from {}".format(manager.latest_checkpoint))

    dqn_agent.update_target()

    episode_rewards = [0.0 for i in range(101)]
    saved_mean_reward = None
    obs_dict = env.reset()
    reset = True

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

        if t % 1000 == 0:
            print('time step ', t)
            print('eps update', exploration.value(t))

        # obs is a dictionary of all agents observations
        action_dict = dqn_agent.choose_action(obs_dict, update_eps=update_eps, **kwargs)
        # action = action[0].numpy()
        # print('action_dict = ', action_dict)
        reset = False
        new_obs_dict, rew, done, _ = env.step_shaped_reward(action_dict)

        if replay_buffer is not None:
            # Store transition in the replay buffer.
            for agent_name in agent_names:
                replay_buffer.add(obs_dict[agent_name], action_dict[agent_name], rew,
                                      new_obs_dict[agent_name], float(done))

        obs = new_obs_dict

        episode_rewards[-1] += rew
        if done:
            obs_dict = env.reset()
            episode_rewards.append(0.0)
            reset = True

        if t > learning_starts and t % train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            td_errors = 0
            for i, agent_name in enumerate(agent_names):
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None

                td_errors += dqn_agent.train(obses_t, actions, rewards, obses_tp1, dones, weights, agent_name)

            if prioritized_replay:
                new_priorities = np.abs(td_errors) + prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)

        if t > learning_starts and t % target_network_update_freq == 0:
            # Update target network periodically.
            dqn_agent.update_target()

        if t % play_test ==0:
            play_test_games(dqn_agent)

        mean_100ep_reward = np.mean(episode_rewards[-101:-1])
        num_episodes = len(episode_rewards)
        if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            print(f'last 100 episode mean reward {mean_100ep_reward} in {num_episodes} playing')
            # logger.record_tabular("steps", t)
            # logger.record_tabular("episodes", num_episodes)
            # logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
            # logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
            # logger.dump_tabular()


import matplotlib.pyplot as plt
import numpy as np

num_agents = 4
render = False
stacked = False
env_name = 'academy_3_vs_1_with_keeper'
channel_dim = (42, 42)
representationType = 'extracted'  # 'extracted': minimaps, 'pixels': raw pixels, 'simple115': vector of 115
env = create_ma_env(num_agents, render, stacked, env_name, representationType, channel_dim)

# obs = env.reset()
# for agent_name in agent_names:
#     obs_a = obs[agent_name]
#     plt.imshow(obs_a[:, :, 3])
#     plt.show()


# action_dict = env.sample()
# print('action_dict ', action_dict)
# obs, rew, done, _ = env.step_(action_dict)
#
# print('obs, rew, done', obs, rew, done)

learn(env)
