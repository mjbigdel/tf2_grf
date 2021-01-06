import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
from deepq.utils import init_env, set_global_seeds
from deepq.learn import Learn


class Config(object):
    def __init__(self):
        self.environment_type = 'gym'  # gym, GFootball, multigrid, minigrid
        self.env_name = 'BreakoutNoFrameskip-v4'  # 'academy_3_vs_1_with_keeper'  academy_empty_goal
        self.num_agents = 1
        self.max_episodes_length = 300
        self.data_path = './plays/'
        self.stacked = False

        # GFootbal Configs
        self.render_train = False
        self.render_test = False
        self.channel_dim = (96 // 2, 72 // 2)
        self.representationType = 'extracted'  # 'extracted': minimaps, 'pixels': raw pixels, 'simple115': vector of 115
        self.train_rewards = 'checkpoints,scoring'
        self.test_rewards = 'scoring'

        # Network Configs
        self.network = 'cnn'
        self.load_path = None
        self.save_path = './models'
        self.num_actions = 1
        self.obs_shape = None
        self.lr = 0.00008
        self.grad_norm_clipping = False
        self.dueling = True
        self.double_q = True
        self.num_extra_data = 2
        self.fp_shape = 0
        self.target_network_update_freq = 10
        self.tau = 0.001

        self.seed = 12
        self.num_timesteps = 100000
        self.batch_size = 32
        self.n_steps = 1
        self.buffer_size = 10000
        self.learning_starts = 10000
        self.print_freq = 10
        self.train_freq = 1
        self.gamma = 0.9
        self.exploration_fraction = 0.4
        self.exploration_final_eps = 0.01
        self.same_reward_for_agents = True
        self.replay_buffer = True
        self.prioritized_replay = False
        self.prioritized_replay_alpha = 0.6
        self.prioritized_replay_beta_iters = None
        self.prioritized_replay_beta0 = 0.4
        self.num_tests = 10
        self.playing_test = 50000

        self.fc1_dims = 512


# main
# if __name__ == 'main':
tf.config.run_functions_eagerly(True)
config = Config()
set_global_seeds(config.seed)

# init env
env = init_env(config, 'train')

config.num_actions = env.action_space.n
config.obs_shape = env.observation_space.shape
config.fp_shape = (config.num_agents - 1) * config.num_actions + config.num_extra_data

dqn_learn = Learn(config, env)
dqn_learn.learn()
