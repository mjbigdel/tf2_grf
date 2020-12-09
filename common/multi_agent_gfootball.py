import numpy as np

'''
Google Football
'''


class GF():
    def __init__(self, num_agents, render, stacked, env_name, representationType, **kwargs):
        self.n_agents = num_agents

        self.env = football_env.create_environment(
            # env_name='test_example_multiagent',
            env_name=env_name,  # env_name='3_vs_GK',
            representation=representationType,
            rewards='scoring',
            stacked=stacked,
            logdir='./tmp/rllib_test',
            write_goal_dumps=True,
            write_full_episode_dumps=True,
            render=render,
            dump_frequency=0,
            number_of_left_players_agent_controls=num_agents,
            channel_dimensions=(42, 42))  # the preferred size for many professional teams' stadiums is 105 by 68 metres

        # channel_dimensions=(3, 3))
        # self.env = wrappers.Simple115StateWrapper(self.env)

        self.episode_limit = 300

        self.obs = None

        self.current_step_num = -1
        # the agent will get reward based on the distance between the ball and targeted goal.
        self.distance_reward = True
        self.discount_on_episode_limit = True
        # how the distance_reward should be discounted,
        # the value of distance_reward is in range [0. , self.distance_reward_discount_factor]
        self.distance_reward_discount_factor = 1.0
        # in order to not punish early shooting, we need to accumulate the reward for all left time steps and
        # reward such player with this reward. Basically,
        # self.accumulate_reward_on_score = builder.config().end_episode_on_score in your scenario.
        self.accumulate_reward_on_score = True
        # TODO:
        self.general_multiplier = 12
        self.owned_by_other_team_reward = -0.4
        self.ball_owned_team = -1  # -1 means no one, 0 means player team
        self.ball_owned_player = -1  # -1 means no one
        # if the ball was owned by the player team, and the ball was passed from one player to the other player,
        # add some punishment. Basically we want the player to be greedy.
        self.pocession_change_reward = -0.1 * (self.episode_limit ** int(self.discount_on_episode_limit))

    def step(self, actions):
        """ Returns reward, terminated, info """
        self.current_step_num += 1
        # len(reward) == num of agents.
        # info stores {'score_reward': int}
        observation, reward, done, info = self.env.step(actions)
        self.obs = observation

        # the customized reward (based on CheckpointRewardWrapper):
        if self.distance_reward:
            observation2 = self.env.unwrapped.observation()
            for rew_index in range(len(reward)):
                o = observation2[rew_index]
                # reward[rew_index] == 1 means that there is a goal from player rew_index. If there is a goal, we
                # add the reward for every player.
                if self.accumulate_reward_on_score and reward[rew_index] == 1:
                    reward[rew_index] += (self.episode_limit - self.current_step_num) / (
                                self.episode_limit ** int(self.discount_on_episode_limit))
                    continue

                if o['ball_owned_team'] != -1 and o['ball_owned_team'] != 0:
                    reward[rew_index] += self.owned_by_other_team_reward / (
                                self.episode_limit ** int(self.discount_on_episode_limit))
                    continue

                if self.ball_owned_player != -1 and self.ball_owned_player != o['ball_owned_player']:
                    self.ball_owned_player = o['ball_owned_player']
                    reward[rew_index] += self.pocession_change_reward / (
                                self.episode_limit ** int(self.discount_on_episode_limit))
                    continue

                # Check if the active player has the ball.
                if ('ball_owned_team' not in o or
                        o['ball_owned_team'] != 0 or
                        'ball_owned_player' not in o or
                        o['ball_owned_player'] != o['active']):
                    continue

                # o['ball'][0] is X, in the range [-1, 1]. o['ball'][1] is Y, in the range [-0.42, 0.42]
                # (2*2+0.42*0.42)**0.5 = 2.0436242316042352
                # the closer d to zero means the closer it is to the (enemy or right???) team's gate
                d = ((o['ball'][0] - 1) ** 2 + o['ball'][1] ** 2) ** 0.5
                # we divide by self.episode_limit since this reward is accumulative, we don't want the accumulative
                # reward to be too large after all.
                reward[rew_index] += (1 - d / 2.0436242316042352) * self.distance_reward_discount_factor / (
                            self.episode_limit ** int(self.discount_on_episode_limit))
                # print("player",rew_index,":",d)
                # print(reward)
                # print(o['ball'])
        # print("o['ball_owned_team']")
        # print(o['ball_owned_team'])
        # print("o['ball_owned_player']")
        # print(o['ball_owned_player'])
        # print("local reward")
        # print(np.sum(reward))
        return np.sum(reward * self.general_multiplier), done, info

    def get_obs(self):
        """ Returns all agent observations in a list """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        obs_agent = self.obs[agent_id].flatten()
        return obs_agent

    def get_obs_size(self):
        """ Returns the shape of the observation """
        # if obs_space is (2, 10, 10, 4) it returns (10, 10, 4)
        obs_size = np.array(self.env.observation_space.shape[1:])
        return int(obs_size.prod())

    def get_state(self):

        # print("Shape of state: %s" % str(state.shape))
        # TODO: difference between observation and state unclear from the google football github
        return self.obs.flatten()
        # return self.env.get_state()

    def get_state_size(self):
        """ Returns the shape of the state"""
        state_size = np.array(self.env.observation_space.shape)
        return int(state_size.prod())

    def get_avail_actions(self):
        """Gives a representation of which actions are available to each agent.
        Returns nested list of shape: n_agents * n_actions_per_agent.
        Each element in boolean. If 1 it means that action is available to agent."""
        # assumed that all actions are available.

        total_actions = self.get_total_actions()

        avail_actions = [[1] * total_actions for i in range(0, self.n_agents)]
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id.
        Returns a list of shape: n_actions of agent.
        Each element in boolean. If 1 it means that action is available to agent."""
        # assumed that all actions are available.
        return [1] * self.get_total_actions()

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take.
        Should be integer of number of actions of an agent. Assumed that all agents have same number of actions."""
        return self.env.action_space.nvec[0]

    def get_stats(self):
        # raise NotImplementedError
        return {}

    def get_agg_stats(self, stats):
        return {}

    def reset(self):
        """ Returns initial observations and states"""
        self.obs = self.env.reset()  # .reshape(self.n_agents)
        self.current_step_num = -1
        # print("Shape of raw observations: %s" % str(self.obs.shape))
        # should be return self.get_obs(), self.get_state()
        return self.get_obs(), self.get_state()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self):
        pass
        # raise NotImplementedError

    def save_replay(self):
        pass
        # raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info


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
"""
import gfootball.env as football_env
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class SingleAgent():
    _counter = 0
    def __init__(self, num_agents, render, stacked, env_name, representationType, channel_dim, rewards, **kwargs):
        self.num_agents = num_agents
        self.id = SingleAgent._counter
        SingleAgent._counter += 1

        self.env = football_env.create_environment(
            env_name=env_name,  # env_name='academy_3_vs_1_with_keeper',
            representation=representationType,
            rewards=rewards,  # checkpoints
            stacked=stacked,
            logdir='./tmp/rllib_test',
            write_goal_dumps=True,
            write_full_episode_dumps=True,
            render=render,
            dump_frequency=0,
            number_of_left_players_agent_controls=1,
            channel_dimensions=channel_dim  # the preferred size for many professional teams' stadiums is 105 by 68 metres
        )
        self.spec = self.Spec(id=self.id)
        self.step = self.env.step
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reset = self.env.reset

        self.episode_limit = 300

        self.current_step_num = -1
        # the agent will get reward based on the distance between the ball and targeted goal.
        self.distance_reward = False
        self.discount_on_episode_limit = True
        # how the distance_reward should be discounted,
        # the value of distance_reward is in range [0. , self.distance_reward_discount_factor]
        self.distance_reward_discount_factor = 1.0
        # in order to not punish early shooting, we need to accumulate the reward for all left time steps and
        # reward such player with this reward. Basically,
        # self.accumulate_reward_on_score = builder.config().end_episode_on_score in your scenario.
        self.accumulate_reward_on_score = True
        # TODO:
        self.general_multiplier = 12
        self.owned_by_other_team_reward = -0.4
        self.ball_owned_team = -1  # -1 means no one, 0 means player team
        self.ball_owned_player = -1  # -1 means no one
        # if the ball was owned by the player team, and the ball was passed from one player to the other player,
        # add some punishment. Basically we want the player to be greedy.
        self.pocession_change_reward = -0.1 * (self.episode_limit ** int(self.discount_on_episode_limit))

    # def step(self, action):
    #     """
    #     :return:
    #         obs as a dict for each agent, reward value as scalar, terminated as True/False, info
    #     """
    #     self.current_step_num += 1
    #     observation, reward, done, info = self.env.step(action)
    #
    #     # if reward == 0:
    #     #     reward = reward - (0.5 / self.episode_limit)
    #
    #     return observation, reward, done, info

    def step_shaped_reward(self, action):
        """
        :return:
            obs as a dict for each agent, reward value as scalar, terminated as True/False, info
        """
        self.current_step_num += 1
        # len(reward) == num of agents.
        # info stores {'score_reward': int}
        # actions = []
        # for key, value in sorted(action_dict.items()):
        #     actions.append(value)
        observation, reward, done, info = self.env.step(action)
        # obs = {}
        # for pos, key in enumerate(sorted(action_dict.keys())):
        #     if self.num_agents > 1:
        #         obs[key] = observation[pos]
        #     else:
        #         obs[key] = observation

        # the customized reward (based on CheckpointRewardWrapper):
        if self.distance_reward:
            observation2 = self.env.unwrapped.observation()
            # for rew_index in range(len(reward)):
            o = observation2[0]
            # print(f'o  is {o}')
            # reward[rew_index] == 1 means that there is a goal from player rew_index. If there is a goal, we
            # add the reward for every player.
            # if self.accumulate_reward_on_score and reward == 1:
            #     reward += (self.episode_limit - self.current_step_num) / (self.episode_limit ** int(self.discount_on_episode_limit))
            #
            # if o['ball_owned_team'] != -1 and o['ball_owned_team'] != 0:
            #     reward += self.owned_by_other_team_reward / (self.episode_limit ** int(self.discount_on_episode_limit))
            #
            # if self.ball_owned_player != -1 and self.ball_owned_player != o['ball_owned_player']:
            #     self.ball_owned_player = o['ball_owned_player']
            #     reward += self.pocession_change_reward / (self.episode_limit ** int(self.discount_on_episode_limit))

            # Check if the active player has the ball.
            # if ('ball_owned_team' not in o or
            #         o['ball_owned_team'] != 0 or
            #         'ball_owned_player' not in o or
            #         o['ball_owned_player'] != o['active']):
            #

            # o['ball'][0] is X, in the range [-1, 1]. o['ball'][1] is Y, in the range [-0.42, 0.42]
            # (2*2+0.42*0.42)**0.5 = 2.0436242316042352
            # the closer d to zero means the closer it is to the (enemy or right???) team's gate
            d = ((o['ball'][0] - 1) ** 2 + o['ball'][1] ** 2) ** 0.5
            # we divide by self.episode_limit since this reward is accumulative, we don't want the accumulative
            # reward to be too large after all.
            reward += (1 - d / 2.0436242316042352) * self.distance_reward_discount_factor / (
                    self.episode_limit ** int(self.discount_on_episode_limit))
            # print("player",rew_index,":",d)
            # print(reward)
            # print(o['ball'])
        # print("o['ball_owned_team']")
        # print(o['ball_owned_team'])
        # print("o['ball_owned_player']")
        # print(o['ball_owned_player'])
        # print("local reward")
        # print(np.sum(reward))
        if reward == 0:
            reward = reward - 0.2
        return observation, reward * self.general_multiplier, done, info

    def close(self):
        self.env.close()

    class Spec:
        def __init__(self, id):
            self.id = id


class RllibGFootball(MultiAgentEnv):
    """An example of a wrapper for GFootball to make it compatible with rllib."""
    _counter = 0
    def __init__(self, num_agents, render, stacked, env_name, representationType, channel_dim, rewards, data_path, **kwargs):
        self.id = RllibGFootball._counter
        RllibGFootball._counter += 1

        self.num_agents = num_agents

        self.env = football_env.create_environment(
            env_name=env_name,  # env_name='academy_3_vs_1_with_keeper',
            representation=representationType,
            rewards=rewards,  # 'scoring, checkpoints'
            stacked=stacked,
            logdir=data_path,
            write_goal_dumps=True,
            write_full_episode_dumps=True,
            render=render,
            dump_frequency=0,
            number_of_left_players_agent_controls=self.num_agents
            # channel_dimensions=channel_dim  # the preferred size for many professional teams' stadiums is 105 by 68 metres
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


        print('self.observation_space = ', self.observation_space)
        print('self.action_space = ', self.action_space)

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
            return original_obs
        else:
            return [original_obs]

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
            return [obs], [reward], done, info

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
