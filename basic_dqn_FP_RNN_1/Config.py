

class Basic_DQN_FP_RNN_1_Conf():
    def __init__(self):
        self.data_path ='./rllib_test/DQN_FP_ConvLSTM/'
        self.load_path = None
        self.save_path = './basic_dqn_FP_RNN_1/models'
        self.num_agents = 2
        self.render_train = False
        self.render_test = False
        self.stacked = False
        self.env_name = 'academy_empty_goal'  # 'academy_3_vs_1_with_keeper'  academy_empty_goal
        self.channel_dim = (48, 36)  # 96,72
        self.representationType = 'extracted'  # 'extracted': minimaps, 'pixels': raw pixels, 'simple115': vector of 115
        self.train_rewards = 'checkpoints,scoring'
        self.test_rewards = 'scoring'
        self.seed = 12
        self.num_actions = 1
        self.obs_shape = None
        self.network = 'resnet_cnn'
        self.num_timesteps = 100000
        self.n_steps = 16
        self.num_tests = 10
        self.replay_buffer = True
        self.batch_size = 32
        self.prioritized_replay = False
        self.prioritized_replay_alpha = 0.6
        self.prioritized_replay_beta_iters = None
        self.prioritized_replay_beta0 = 0.4
        self.exploration_fraction = 0.1
        self.exploration_final_eps = 0.1
        self.same_reward_for_agents = False
        self.gamma = 0.99
        self.buffer_size = 5000
        self.learning_starts = 1000
        self.print_freq = 50
        self.train_freq = 1
        self.playing_test = 500
        self.grad_norm_clipping = False
        self.lr = 0.1
        self.dueling = True
        self.double_q = True
        self.num_extra_data = 2
        self.target_network_update_freq = 10
        self.tau = 0.001



