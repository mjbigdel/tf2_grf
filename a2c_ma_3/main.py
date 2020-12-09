import tensorflow as tf
tf.config.run_functions_eagerly(True)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Mute missing instructions errors
import numpy as np
MODEL_PATH = './models'
SEED = 0  # np.random.randint(1,100,1)[0]
print(f'SEED is {SEED}')
from ray.tune.registry import register_env

from common.subproc_vec_env import SubprocVecEnv
from common.utils import create_env, create_ma_env

from a2c_ma_3.a2c import learn, play_test
from a2c_ma_3.Config import Conf

def train():
    conf = Conf()
    num_timesteps = conf.num_timesteps
    single_agent = conf.single_agent
    num_agents = conf.num_agents
    render = conf.render
    stacked = conf.stacked
    env_name = conf.env_name
    channel_dim = conf.channel_dim
    representationType = conf.representationType
    rewards = 'checkpoints,scoring'  # checkpoints,

    def make_env(rank):
        def _thunk():
            if single_agent:
                env = create_env(num_agents, render, stacked, env_name, representationType, channel_dim, rewards)
            else:
                # register_env('gfootball', lambda _: RllibGFootball(num_agents))
                env = create_ma_env(num_agents, render, stacked, env_name, representationType, channel_dim, rewards)
            # env = gym.make('gfootball:GFootball-11_vs_11_easy_stochastic-SMM-v0',
            #                stacked=True)
            env.env.seed(SEED + rank)
            return env
        return _thunk

    # env = SubprocVecEnv([make_env(i) for i in range(conf.nenv)])
    env = create_ma_env(num_agents, render, stacked, env_name, representationType, channel_dim, rewards)
    a2c_agent = None
    network = conf.network

    a2c_agent = learn(single_agent, network, env, num_agents, seed=SEED, nsteps=conf.nsteps,
                      total_timesteps=num_timesteps, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
                      lr=0.00008, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99,
                      log_interval=10, load_path=None, save_path=MODEL_PATH,
                      shared_weights=True)
    env.close()
    return a2c_agent


def main():
    os.makedirs(MODEL_PATH, exist_ok=True)
    a2c_agent = train()
    play_test(a2c_agent)


if __name__ == "__main__":
    main()

