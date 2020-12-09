

class Basic_DQN_Conf():
    def __init__(self):
        self.data_path ='./rllib_test/DQN/'
        self.load_path = None
        self.save_path = None
        self.num_agents = 2
        self.render_train = False
        self.render_test = False
        self.stacked = True
        self.env_name = 'academy_run_to_score'  # 'academy_3_vs_1_with_keeper'
        self.channel_dim = (51, 40)
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



