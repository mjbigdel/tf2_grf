

class Conf():
    def __init__(self):
        self.single_agent = False
        self.num_agents = 3
        self.render = False
        self.stacked = True
        self.env_name = 'academy_3_vs_1_with_keeper'  # 'academy_3_vs_1_with_keeper'
        self.channel_dim = (51, 40)
        self.representationType = 'extracted'  # 'extracted': minimaps, 'pixels': raw pixels, 'simple115': vector of 115
        self.network = 'gfootball_impala_cnn'
        self.num_timesteps = 1000000
        self.nenv = 8
        self.nsteps = 32