
import time
from collections import deque
import numpy as np

from common import logger
from common.utils import set_global_seeds, explained_variance, safemean
from common.tf2_models import get_network_builder, build_cnn, build_mlp,\
    build_cnn_rnn, build_ma_cnn, gfootball_impala_cnn, gfootball_impala_cnn_rnn
from common.utils import create_env, create_ma_env


from a2c_ma_3.utils import *
from a2c_ma_3.a2c_agent import MAgent, Agent
from a2c_ma_3.runner import Runner, MRunner
from a2c_ma_3.Config import Conf


# def play_test_dict(a2c_agent):
#     num_agents = 4
#     render = True
#     stacked = True
#     env_name = 'academy_3_vs_1_with_keeper'
#     channel_dim = (42, 42)
#     representationType = 'extracted'
#     env = create_ma_env(num_agents, render, stacked, env_name, representationType, channel_dim)
#
#     import numpy as np
#     num_tests = 3
#     rewards = np.zeros(num_tests)
#     for t in range(num_tests):
#         obs = env.reset()
#         state = a2c_agent.initial_state if hasattr(a2c_agent, 'initial_state') else None
#         done = False
#         while not done:
#             action_dict = {}
#             if state is not None:
#                 actions, _, state, _ = a2c_agent.step(obs)
#             else:
#                 obs_ = tf.constant([obs[agent_name] for agent_name in env.agent_names()])
#                 actions, _, _, _ = a2c_agent.step(obs_)
#                 for a, agent_name in enumerate(env.agent_names()):
#                     action_dict[agent_name] = actions[a]
#             obs1, rew, done, _ = env.step(action_dict)
#             obs = obs1
#             done = done['__all__']
#             if done:
#                 print(f'rew for test {t} is {rew}')
#                 rewards[t] = rew['agent_00']
#
#     print(f'mean reward is {np.mean(rewards)}')


def play_test(a2c_agent):
    conf = Conf()
    single_agent = conf.single_agent
    num_agents = conf.num_agents
    stacked = conf.stacked
    env_name = conf.env_name
    channel_dim = conf.channel_dim
    representationType = conf.representationType
    rewards = 'scoring'
    render = True

    if single_agent:
        env = create_env(num_agents, render, stacked, env_name, representationType, channel_dim, rewards)
    else:
        env = create_ma_env(num_agents, render, stacked, env_name, representationType, channel_dim, rewards)

    num_tests = 5
    rewards = np.zeros(num_tests)
    for t in range(num_tests):
        obs = env.reset()
        state = a2c_agent.initial_state if hasattr(a2c_agent, 'initial_state') else None
        done = False
        rews = []
        while not done:
            action_list = []
            if single_agent:
                obs = tf.constant([obs])
                action, _, state, _ = a2c_agent.step(obs)
                action_list = action.numpy()
            else:
                for a in env.agent_ids():
                    obs_ = tf.constant([obs[a]])
                    action, _, _, _ = a2c_agent.step(obs_, a)
                    action_list.append(action.numpy()[0])
            # print(action_list)
            obs1, rew, done, _ = env.step(action_list)
            obs = obs1
            rews.append(rew)
            if done:
                print(f'sum rew for test {t} is {np.sum(np.asarray(rews))}')
                if single_agent:
                    rewards[t] = rew
                else:
                    rewards[t] = rew[0]

    print(f'mean reward is {np.mean(rewards)}')

    env.close()


def learn(
    single_agent,
    network,
    env,
    num_agents,
    seed=None,
    nsteps=5,
    total_timesteps=int(80e6),
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    lr=7e-4,
    lrschedule='linear',
    epsilon=1e-5,
    alpha=0.99,
    gamma=0.99,
    log_interval=100,
    load_path=None,
    save_path=None,
    shared_weights=True,
    **network_kwargs):

    """
    Main entrypoint for A2C algorithm. Train a policy with given network architecture on a given environment using a2c algorithm.

    Parameters:
    -----------

    network:            policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                        specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                        tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                        neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                        See baselines.common/policies.py/lstm for more details on using recurrent nets in policies


    env:                RL environment. Should implement interface similar to VecEnv (baselines.common/vec_env) or be wrapped with DummyVecEnv (baselines.common/vec_env/dummy_vec_env.py)


    seed:               seed to make random number sequence in the alorightm reproducible. By default is None which means seed from system noise generator (not reproducible)

    nsteps:             int, number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                        nenv is number of environment copies simulated in parallel)

    total_timesteps:    int, total number of timesteps to train on (default: 80M)

    vf_coef:            float, coefficient in front of value function loss in the total loss function (default: 0.5)

    ent_coef:           float, coeffictiant in front of the policy entropy in the total loss function (default: 0.01)

    max_gradient_norm:  float, gradient is clipped to have global L2 norm no more than this value (default: 0.5)

    lr:                 float, learning rate for RMSProp (current implementation has RMSProp hardcoded in) (default: 7e-4)

    lrschedule:         schedule of learning rate. Can be 'linear', 'constant', or a function [0..1] -> [0..1]
                        that takes fraction of the training progress as input and
                        returns fraction of the learning rate (specified as lr) as output

    epsilon:            float, RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update)
                        (default: 1e-5)

    alpha:              float, RMSProp decay parameter (default: 0.99)

    gamma:              float, reward discounting parameter (default: 0.99)

    log_interval:       int, specifies how frequently the logs are printed out (default: 100)

    **network_kwargs:   keyword arguments to the policy / network builder. See b
                        aselines.common/policies.py/build_policy and arguments to a particular type of network
                        For instance, 'mlp' network architecture has arguments num_hidden and num_layers.

    """

    set_global_seeds(seed)
    total_timesteps = int(total_timesteps)
    # Get the nb of env
    nenvs = env.num_envs
    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    if network == 'gfootball_impala_cnn':
        policy_network = gfootball_impala_cnn(ob_space.shape, ac_space.n, num_agents,
                     512, 512, shared_weights, 'shared_impala_network')
    elif network == 'cnn':
        policy_network = build_ma_cnn(ob_space.shape, ac_space.n, num_agents,
                                      512, 512, shared_weights, 'shared_q_network')
    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nupdates = total_timesteps // nbatch

    # Instantiate the model object (that creates step_model and train_model)
    if single_agent:
        a2c_agent = Agent(ac_space=ac_space, policy_network=policy_network, nupdates=nupdates, ent_coef=ent_coef,
                       vf_coef=vf_coef, max_grad_norm=max_grad_norm, lr=lr, alpha=alpha,
                       epsilon=epsilon, total_timesteps=total_timesteps)
        # Instantiate the runner object
        runner = Runner(env, a2c_agent, nsteps=nsteps, gamma=gamma)
    else:
        a2c_agent = MAgent(agent_id=1, num_agents=num_agents, ac_space=ac_space, policy_network=policy_network,
                           nupdates=nupdates, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                           lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps)
        # Instantiate the runner object
        runner = MRunner(env, a2c_agent, nsteps=nsteps, gamma=gamma, num_agents=num_agents)

    if load_path is not None:
        print(f'models loaded from {load_path}')
        a2c_agent.load(load_path)


    epinfobuf = deque(maxlen=100)

    # Start total timer
    tstart = time.time()

    for update in range(1, nupdates+1):
        # Get mini batch of experiences
        obs, states, rewards, masks, actions, values, epinfos = runner.run()
        epinfobuf.extend(epinfos)
        if single_agent:
            obs = tf.constant(obs[:, ::])
            # print(obs_.shape)
            if states is not None:
                states = tf.constant(states)
            rewards = tf.constant(rewards[:])
            actions = tf.constant(actions[:])
            values = tf.constant(values[:])
            # print('shapes are ', obs_.shape, rewards_.shape, actions_.shape, values_.shape)
            # print('dtypes are ', obs_.dtype, rewards_.dtype, actions_.dtype, values_.dtype)
            masks = tf.constant(masks)
            policy_loss, value_loss, policy_entropy = a2c_agent.train(
                obs, states, rewards, masks, actions, values)
        else:
            # print('obs[0][agent_00] is ', [obs[k][runner.agent_names[0]] for k in range(nbatch)])
            for i in range(num_agents):
                obs_ = tf.constant(obs[:,i,::])
                # print(obs_.shape)
                if states is not None:
                    states = tf.constant(states)
                rewards_ = tf.constant(rewards[:,i])
                actions_ = tf.constant(actions[:,i])
                values_ = tf.constant(values[:,i])
                # print('shapes are ', obs_.shape, rewards_.shape, actions_.shape, values_.shape)
                # print('dtypes are ', obs_.dtype, rewards_.dtype, actions_.dtype, values_.dtype)
                masks = tf.constant(masks)
                policy_loss, value_loss, policy_entropy = a2c_agent.train(
                    obs_, states, rewards_, masks, actions_, values_, i)

        nseconds = time.time()-tstart
        # Calculate the fps (frame per second)
        fps = int((update*nbatch)/nseconds)

        if update % (log_interval*10) == 0:
            print(f'models saved at {save_path}')
            a2c_agent.save(save_path)

        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            if single_agent:
                ev = explained_variance(values, rewards)
                logger.record_tabular("rewards", np.mean(rewards))
            else:
                ev = explained_variance(values[:,0], rewards[:,0])
                logger.record_tabular("rewards", np.mean(rewards[:,0]))

            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            # logger.record_tabular("policy_entropy", float(K.eval(policy_entropy)))
            # logger.record_tabular("value_loss", float(K.eval(value_loss)))
            logger.record_tabular("explained_variance", float(ev))
            # logger.record_tabular("eprewmean", safemean([epinfo['r'] for epinfo in epinfobuf]))
            # logger.record_tabular("eplenmean", safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.dump_tabular()

        if update % (log_interval*20) == 0:
            play_test(a2c_agent)

    return a2c_agent

