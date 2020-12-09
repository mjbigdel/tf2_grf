import tensorflow as tf
import configparser

from DQN_FP.utils import init_env, init_dir, init_log, copy_file
from DQN_FP.nstep_dqn_fp import learn

def train():
    tf.config.run_functions_eagerly(True)
    base_dir = './rl_test'
    config_dir = './config.ini'
    dirs = init_dir(base_dir)
    init_log(dirs['log'])
    copy_file(config_dir, dirs['data'])
    config = configparser.ConfigParser()
    config.read(config_dir)

    # init env
    env = init_env(config['ENV_CONFIG'], 'train')

    learn(env, config, 0)







train()

