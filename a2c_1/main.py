import tensorflow as tf
tf.config.run_functions_eagerly(True)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from common.subproc_vec_env import SubprocVecEnv
from common.utils import create_env, create_ma_env
from common.multi_agent_gfootball import RllibGFootball
import os
from a2c_1.a2c import learn, play_test
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Mute missing instructions errors
import gym
from common.tf2_models import build_ma_cnn
import numpy as np
MODEL_PATH = './models'
SEED = np.random.randint(1,100,1)[0]
print(f'SEED is {SEED}')
from ray.tune.registry import register_env


def train(num_timesteps, num_cpu):
    # Simple environment with `num_agents` independent players
    num_agents = 4
    render = False
    stacked = True
    env_name = 'academy_run_to_score'  # 'academy_3_vs_1_with_keeper'
    channel_dim = (42, 42)
    representationType = 'extracted'  # 'extracted': minimaps, 'pixels': raw pixels, 'simple115': vector of 115
    register_env('gfootball', lambda _: RllibGFootball(num_agents))
    def make_env(rank):
        def _thunk():
            env = create_env(num_agents, render, stacked, env_name, representationType, channel_dim)
            # env = gym.make('gfootball:GFootball-11_vs_11_easy_stochastic-SMM-v0',
            #                stacked=True)
            env.env.seed(SEED + rank)
            return env
        return _thunk

    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    obs = env.reset()
    # print(obs)
    # learn(env, SEED, total_timesteps=int(num_timesteps * 1.1))
    network = 'cnn'
    a2c_agent = learn(network, env, seed=None, nsteps=4, total_timesteps=num_timesteps,
          vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear',
          epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=10, load_path=None
          )
    env.close()

    return a2c_agent


def main():
    os.makedirs(MODEL_PATH, exist_ok=True)
    steps = 1000000
    nenv = 32
    a2c_agent = train(steps, num_cpu=nenv)
    play_test(a2c_agent)



if __name__ == "__main__":
    main()
