import os
import logging
import subprocess
import time
import configparser
import numpy as np

from common.utils import create_ma_env

def check_dir(cur_dir):
    if not os.path.exists(cur_dir):
        return False
    return True


def copy_file(src_dir, tar_dir):
    cmd = 'cp %s %s' % (src_dir, tar_dir)
    subprocess.check_call(cmd, shell=True)


def find_file(cur_dir, suffix='.ini'):
    for file in os.listdir(cur_dir):
        if file.endswith(suffix):
            return cur_dir + '/' + file
    logging.error('Cannot find %s file' % suffix)
    return None


def init_dir(base_dir, pathes=['log', 'data', 'model']):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dirs = {}
    for path in pathes:
        cur_dir = base_dir + '/%s/' % path
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
    return dirs


def init_log(log_dir):
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO,
                        handlers=[logging.FileHandler('%s/%d.log' % (log_dir,
                                  time.time())), logging.StreamHandler() ])

def init_env(config, mode='train'):
    data_path = config.get('data_path')
    num_agents = config.getint('num_agents')
    render = config.getboolean('render')
    stacked = config.getboolean('stacked')
    env_name = config.get('env_name')
    channel_dim = config.get('channel_dim')
    representationType = config.get('representationType')
    train_rewards = config.get('train_rewards')
    test_rewards = config.get('test_rewards')
    if mode == 'train':
        rewards = train_rewards
    else:
        rewards = test_rewards
        render = True
    return create_ma_env(num_agents, render, stacked, env_name, representationType, channel_dim,
                         rewards, data_path)


def play_test_games(dqn_agent, config):
    import tensorflow as tf
    num_tests = config.getint('ENV_CONFIG', 'num_tests')
    test_env = init_env(config['ENV_CONFIG'], 'test')

    test_rewards = np.zeros(num_tests)
    for i in range(num_tests):
        test_done = False
        test_obs_all = test_env.reset()
        # print(np.asarray(test_obs_all).shape)
        while not test_done:
            test_obs_all = tf.constant(test_obs_all)
            test_action_list = dqn_agent.choose_greedy_action(test_obs_all)
            test_new_obs_list, test_rew_list, test_done, _ = test_env.step(test_action_list)
            test_obs_all = test_new_obs_list

            if test_done:
                print(f'test_reward_dict for test {i} is {test_rew_list}')
                test_rewards[i] = np.mean(test_rew_list)

    print(f'mean reward of {num_tests} tests is {np.mean(test_rewards)}')
    test_env.close()

