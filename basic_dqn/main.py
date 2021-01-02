
import tensorflow as tf

from basic_dqn.agents import Agent
from common.utils import set_global_seeds, init_env

# extracted as representationType:
# The first plane P holds the position of players on the left team, P[y,x]
# is 255 if there is a player at position (x,y), otherwise, its value is 0.
# The second plane holds in the same way the position of players on the right team.
# The third plane holds the position of the ball. The last plane holds the active player.

class Basic_DQN_Conf():
    def __init__(self):
        self.data_path ='./rllib_test/DQN/'
        self.load_path = None
        self.save_path = None
        self.num_agents = 2
        self.render_train = False
        self.render_test = False
        self.stacked = False
        self.env_name = 'academy_empty_goal_close'  # 'academy_3_vs_1_with_keeper'
        self.channel_dim = (96, 72)
        self.representationType = 'extracted'  # 'extracted': minimaps, 'pixels': raw pixels, 'simple115': vector of 115
        self.train_rewards = 'checkpoints,scoring'
        self.test_rewards = 'scoring'
        self.seed = 12
        self.num_actions = 1
        self.obs_shape = None
        self.network = 'resnet_cnn'
        self.num_timesteps = 100000
        self.n_steps = 16
        self.num_tests = 5
        self.replay_buffer = True
        self.batch_size = 8
        self.prioritized_replay = False
        self.prioritized_replay_alpha = 0.6
        self.prioritized_replay_beta_iters = None
        self.prioritized_replay_beta0 = 0.4
        self.exploration_fraction = 0.2
        self.exploration_final_eps = 0.01
        self.same_reward_for_agents = True
        self.gamma = 0.99
        self.target_network_update_freq = 1000
        self.buffer_size = 1000
        self.learning_starts = 200
        self.print_freq = 10
        self.train_freq = 1
        self.playing_test = 100
        self.grad_norm_clipping = False
        self.lr = 0.00008
        self.dueling = False
        self.double_q = True


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


# # train()
# tf.config.run_functions_eagerly(True)
# config = Basic_DQN_Conf()
# # init env
# env = init_env(config, 'train')
#
# print(f'env.observation_space.shape {env.observation_space.shape}')
#
# obs = env.reset()
# print(obs.shape)
# print(obs.dtype)
# import matplotlib.pyplot as plt
#
# plt.imshow(obs[1, :, :, 0])
# plt.show()