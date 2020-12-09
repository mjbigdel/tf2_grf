
import tensorflow as tf

from basic_dqn.agents import Agent
from basic_dqn.Config import Basic_DQN_Conf
from common.utils import set_global_seeds
from basic_dqn.utils import init_env

def init_agent(config, env):
    agent = Agent(config, env)
    return agent

def train():

    tf.config.run_functions_eagerly(True)
    config = Basic_DQN_Conf()
    set_global_seeds(config.seed)

    # init env
    env = init_env(config, 'train')

    config.num_actions = env.action_space.n
    config.obs_shape = env.observation_space.shape

    agent = init_agent(config, env)
    agent.learn()


train()

