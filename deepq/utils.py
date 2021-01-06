import tensorflow as tf
import numpy as np
import random


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1. - done)  # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]


def create_ma_env(num_agents, render, stacked, env_name, representationType, channel_dim, rewards, data_path):
    from common.multi_agent_gfootball import RllibGFootball
    return RllibGFootball(num_agents, render, stacked, env_name, representationType, channel_dim, rewards, data_path)


def init_env(config, mode='train'):
    data_path = config.data_path
    num_agents = config.num_agents
    stacked = config.stacked
    env_name = config.env_name

    if config.environment_type == 'GFootball':
        channel_dim = config.channel_dim
        representationType = config.representationType
        train_rewards = config.train_rewards
        test_rewards = config.test_rewards

        if mode == 'train':
            rewards = train_rewards
            render = config.render_train
        else:
            rewards = test_rewards
            render = config.render_test
        return create_ma_env(num_agents, render, stacked, env_name, representationType,
                             channel_dim, rewards, data_path)

    if config.environment_type == 'multigrid':
        from common.multi_agent_gfootball import MultiGrid
        from common.multi_agent_gfootball import MultiGrid
        return MultiGrid('soccer')

    if config.environment_type == 'minigrid':
        from common.multi_agent_gfootball import MiniGrid
        return MiniGrid('MiniGrid-Empty-5x5-v0')

    if config.environment_type == 'gym':
        NotImplementedError
        # Todo


@tf.function
def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.math.abs(x) < delta,
        tf.math.square(x) * 0.5,
        delta * (tf.math.abs(x) - 0.5 * delta)
    )


def set_global_seeds(i):
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    myseed = i + 1000 * rank if i is not None else None
    try:
        import tensorflow as tf
        tf.random.set_seed(myseed)
    except ImportError:
        pass
    np.random.seed(myseed)
    random.seed(myseed)




