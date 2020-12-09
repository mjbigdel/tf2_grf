
from common.utils import create_ma_env

def init_env(config, mode='train'):
    data_path = config.data_path
    num_agents = config.num_agents
    stacked = config.stacked
    env_name = config.env_name
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
    return create_ma_env(num_agents, render, stacked, env_name, representationType, channel_dim,
                         rewards, data_path)

