import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
from distdeepq_FullEpisode.utils import init_env, set_global_seeds, register_multi_grid_env
from distdeepq_FullEpisode.learn import Learn


class Config:
    def __init__(self):
        # gym -> 'BreakoutNoFrameskip-v4', ...
        # GFootball -> 'academy_3_vs_1_with_keeper'  'academy_empty_goal', ...
        # multigrid -> 'soccer', ...
        # minigrid -> 'MiniGrid-Empty-5x5-v0', ...
        self.environment_type = 'GFootball'  # gym, GFootball, multigrid, minigrid
        self.env_name = 'academy_empty_goal_close'  #
        self.num_agents = 2
        self.max_episodes_length = 100
        self.num_episodes = 10000
        self.data_path = './plays/'
        self.stacked = False

        # GFootbal Configs
        self.render_train = False
        self.render_test = False
        self.channel_dim = (96//2, 72//2)
        self.representationType = 'extracted'  # 'extracted': minimaps, 'pixels': raw pixels, 'simple115': vector of 115
        self.train_rewards = 'checkpoints,scoring'
        self.test_rewards = 'scoring'

        # Network Configs
        self.network = 'cnn'  # 'impala_cnn', 'cnn', 'mlp'
        self.load_path = None
        self.save_path = './models'
        self.num_actions = 1
        self.obs_shape = None
        self.lr = 0.00008
        self.grad_norm_clipping = False
        self.num_extra_data = 2
        self.fp_shape = 0
        self.target_network_update_freq = 10
        self.tau = 0.001
        self.atoms = 16
        self.dueling = True
        self.double_q = True
        self.distributionalRL = True
        self.noisy_layers = True

        self.seed = 12
        self.num_timesteps = 1000000
        self.batch_size = 8
        self.n_steps = 32
        self.buffer_size = 1000
        self.learning_starts = 10
        self.print_freq = 10
        self.train_freq = 1
        self.gamma = 0.99
        self.exploration_fraction = 0.3
        self.exploration_final_eps = 0.01
        self.same_reward_for_agents = True
        self.replay_buffer = True
        self.prioritized_replay = True
        self.prioritized_replay_alpha = 0.6
        self.prioritized_replay_beta_iters = None
        self.prioritized_replay_beta0 = 0.4
        self.prioritized_replay_eps = 1e-6
        self.num_tests = 10
        self.playing_test = 100

        self.conv_layers = [(32, 8, 4), (16, 4, 2), (16, 2, 1)]  # (filters, kernel, stride)
        self.impala_layers = [(16, 2), (32, 2), (32, 2), (32, 2)]  # (filters, kernel, stride)
        self.fc1_dims = 256
        self.fc2_dims = 256
        self.rnn_dim = 512


# main
# if __name__ == 'main':
tf.config.run_functions_eagerly(True)
config = Config()
set_global_seeds(config.seed)

# init env
if config.environment_type == 'multigrid':
    register_multi_grid_env(config.env_name)
env = init_env(config, 'train')

config.num_actions = env.action_space.n
config.obs_shape = env.observation_space.shape
config.fp_shape = (config.num_agents - 1) * config.num_actions + config.num_extra_data

dqn_learn = Learn(config, env)
dqn_learn.learn()
