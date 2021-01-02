# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simple example of setting up a multi-agent version of GFootball with rllib.
    with some modifications
"""
import gfootball.env as football_env
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np

class RllibGFootball(MultiAgentEnv):
    """An example of a wrapper for GFootball to make it compatible with rllib."""
    _counter = 0
    def __init__(self, num_agents, render, stacked, env_name, representationType, channel_dim, rewards, data_path, **kwargs):
        self.id = RllibGFootball._counter
        RllibGFootball._counter += 1

        self.num_agents = num_agents

        self.env = football_env.create_environment(
            env_name=env_name,
            representation=representationType,
            rewards=rewards,
            stacked=stacked,
            logdir=data_path,
            write_goal_dumps=True,
            write_full_episode_dumps=True,
            render=render,
            dump_frequency=0,
            number_of_left_players_agent_controls=self.num_agents,
            channel_dimensions=channel_dim  # the preferred size for many professional teams' stadiums is 105 by 68 metres
        )

        self.reset = self.reset_list
        self.step = self.step_list
        self.sample = self.sample_list

        self.spec = self.Spec(self.id)

        if num_agents > 1:
            self.action_space = gym.spaces.Discrete(self.env.action_space.nvec[1])
            self.observation_space = gym.spaces.Box(
                                                low=self.env.observation_space.low[0],
                                                high=self.env.observation_space.high[0],
                                                dtype=self.env.observation_space.dtype)
        else:
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space

        print('self.observation_space.shape = ', self.observation_space.shape)
        print('self.action_space.n = ', self.action_space.n)

    def unwrapped_obs(self):
        return self.env.unwrapped.observation()

    def agent_ids(self):
        return [i for i in range(self.num_agents)]

    def reset_list(self):
        """
        :return:
            obs as a list for each agent
        """
        original_obs = self.env.reset()
        if self.num_agents > 1:
            return np.array(original_obs, copy=False)
        else:
            return np.expand_dims(np.array(original_obs, copy=False), 0)  # adds dimension to support single agent case

    def step_list(self, action_list):
        """
        :return:
            obs as a list for each agent
            reward as a list for each agent,
            done as a True/False value, info
        """

        obs, reward, done, info = self.env.step(action_list)
        if self.num_agents > 1:
            return obs, reward, done, info
        else:
            return np.expand_dims(np.array(obs, copy=False), 0), np.expand_dims(reward, 0), done, info

    def close(self):
        self.env.close()

    def sample_list(self):
        """

        :return: list contain action for each agent
        """
        return self.env.action_space.sample()

    class Spec:
        def __init__(self, id):
            self.id = id


import gym
from gym.envs.registration import register
from gym_minigrid.wrappers import FlatObsWrapper
class MultiGrid():
    """An example of a wrapper for GFootball to make it compatible with rllib."""
    _counter = 0
    def __init__(self, env_name, **kwargs):
        self.id = MultiGrid._counter
        MultiGrid._counter += 1

        # self.num_agents = num_agents

        if env_name == 'soccer':
            register(
                id='multigrid-soccer-v0',
                entry_point='gym_multigrid.envs:SoccerGame4HEnv10x15N2',
            )
            self.env = gym.make('multigrid-soccer-v0')

        else:
            register(
                id='multigrid-collect-v0',
                entry_point='gym_multigrid.envs:CollectGame4HEnv10x10N2',
            )
            self.env = gym.make('multigrid-collect-v0')

        self.num_agents = len(self.env.agents)

        # self.env = football_env.create_environment(
        #     env_name=env_name,
        #     representation=representationType,
        #     rewards=rewards,
        #     stacked=stacked,
        #     logdir=data_path,
        #     write_goal_dumps=True,
        #     write_full_episode_dumps=True,
        #     render=render,
        #     dump_frequency=0,
        #     number_of_left_players_agent_controls=self.num_agents,
        #     channel_dimensions=channel_dim  # the preferred size for many professional teams' stadiums is 105 by 68 metres
        # )

        self.reset = self.reset_list
        self.step = self.step_list
        self.sample = self.sample_list

        self.spec = self.Spec(self.id)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        print('self.observation_space = ', self.observation_space)
        print('self.action_space = ', self.action_space)

    def agent_ids(self):
        return [i for i in range(self.num_agents)]

    def reset_list(self):
        """
        :return:
            obs as a list for each agent
        """
        original_obs = self.env.reset()
        if self.num_agents > 1:
            return np.array(original_obs, copy=False)
        else:
            return np.expand_dims(np.array(original_obs, copy=False), 0)  # adds dimension to support single agent case

    def step_list(self, action_list):
        """
        :return:
            obs as a list for each agent
            reward as a list for each agent,
            done as a True/False value, info
        """

        obs, reward, done, info = self.env.step(action_list)
        if self.num_agents > 1:
            return obs, reward, done, info
        else:
            return np.expand_dims(np.array(obs, copy=False), 0), np.expand_dims(reward, 0), done, info

    def close(self):
        self.env.close()

    def sample_list(self):
        """

        :return: list contain action for each agent
        """
        return self.env.action_space.sample()

    class Spec:
        def __init__(self, id):
            self.id = id
#
# data_path = './rllib_test/DQN/'
# single_agent = True
# num_agents = 2
# render = False
# stacked = True
# env_name = 'academy_empty_goal'
# channel_dim = (51, 40)
# representationType = 'extracted'
# train_rewards = 'checkpoints,scoring'
# test_rewards = 'checkpoints, scoring'
#
#
# env = RllibGFootball(num_agents, render, stacked, env_name, representationType, channel_dim, train_rewards, data_path)
# obs = env.reset()
# print(np.array(obs).shape)
# action_list = env.sample()
# print(action_list)
#
# obs1, rewards, dones, infos = env.step(action_list)
# print(np.array(obs1).shape)
# print(rewards)
# print(dones)
# print(infos)
#

# env = MultiGrid('soccer')
# obs = env.reset()
# print(f'obs.shape {obs.shape}')
# print(f'obs[0], {obs[0]}')
# print(f'env.num_agents, {env.num_agents}')
# action = env.sample()
# print(action)
# x = env.step(action)
#
# print(x)