
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from basic_dqn_FP.agents import Agent
from common.utils import set_global_seeds, init_env


class Basic_DQN_FP_Conf():
    def __init__(self):
        self.data_path ='./rllib_test/DQN_FP_2/'
        self.load_path = None
        self.save_path = './basic_dqn_FP/models'
        self.num_agents = 2
        self.render_train = False
        self.render_test = False
        self.stacked = False
        self.env_name = 'academy_empty_goal_close'  # 'academy_3_vs_1_with_keeper'  academy_empty_goal
        self.channel_dim = (96//2, 72//2)
        self.representationType = 'extracted'  # 'extracted': minimaps, 'pixels': raw pixels, 'simple115': vector of 115
        self.train_rewards = 'checkpoints,scoring'
        self.test_rewards = 'scoring'
        self.seed = 12
        self.num_actions = 1
        self.obs_shape = None
        self.network = 'tdcnn_rnn'
        self.num_timesteps = 10000
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
        self.gamma = 0.995
        self.buffer_size = 1000
        self.learning_starts = 1000
        self.print_freq = 10
        self.train_freq = 1
        self.playing_test = 500
        self.grad_norm_clipping = False
        self.lr = 0.000011
        self.dueling = True
        self.double_q = True
        self.add_fps = False
        self.num_extra_data = 2
        self.fp_shape = 0
        self.target_network_update_freq = 10
        self.tau = 0.001


def init_agent(config, env):
    agent = Agent(config, env)
    return agent


def train():
    tf.config.run_functions_eagerly(True)
    config = Basic_DQN_FP_Conf()
    set_global_seeds(config.seed)

    # init env
    env = init_env(config, 'train')
    config.num_actions = env.action_space.n
    config.obs_shape = env.observation_space.shape
    if config.num_agents > 1:
        config.add_fps = True
        config.fp_shape = (config.num_agents - 1) * config.num_actions + config.num_extra_data

    agent = init_agent(config, env)
    agent.learn()


train()

