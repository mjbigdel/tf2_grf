from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import ray
from ray.tune.registry import register_env

from envs import RllibGFootball
from simple_dqn import Agent
from utils import plotLearning

parser = argparse.ArgumentParser()
parser.add_argument('--num-agents', type=int, default=3)
parser.add_argument('--num-policies', type=int, default=2)
parser.add_argument('--num-iters', type=int, default=100000)
parser.add_argument('--simple', action='store_true')


def create_action_dict(agent_names, default_action):
    """ input:
            list of agent names
        output:
            dictionary with agent name as key and default_action as value.
    """
    defualt_actions = [default_action for i in range(len(agent_names))]
    action_dict = dict(zip(agent_names, defualt_actions))

    return action_dict


def sample_actions(action_dict):
    """ input:
            dictionary of agents and there actions
        output:
            dictionary with agent name as key and random sampled action as value.
    """
    for pos, key in enumerate(sorted(action_dict.keys())):
        action_dict[key] = single_env.action_space.sample()

    return action_dict


def _init_agents(agent_names, shared_weights):
    if shared_weights:
        NotImplementedError
    else:
        agents = []
        for i in range(len(agent_names)):
            agents.append(Agent(lr=0.0005, gamma=0.99, n_actions=n_actions, epsilon=1.0,
                                batch_size=batch_size, input_dims=state_dims))
        agents_dict = dict(zip(agent_names, agents))

    return agents_dict


def _init_choose_action_agent(agent_name, agent_obs):
    return agents_dict[agent_name].choose_action(agent_obs)


if __name__ == '__main__':
    args = parser.parse_args()
    ray.init(num_gpus=1)
    batch_size = 8

    # Simple environment with `num_agents` independent players
    register_env('gfootball', lambda _: RllibGFootball(args.num_agents))
    single_env = RllibGFootball(args.num_agents)
    obs_space = single_env.observation_space  # Box(0, 255, (42, 42, 4), uint8)
    act_space = single_env.action_space  # Discrete(19)

    state_dims = single_env.observation_space.shape
    print('state_dims is ', state_dims)
    n_actions = single_env.action_space.n
    print('n_actions is ', n_actions)

    observation = single_env.reset()
    agent_names = observation.keys()
    shared_weights = False

    action_dict = create_action_dict(agent_names=agent_names, default_action=0)
    print('action_dict is ', action_dict)

    # obs, rew, done, info = single_env.step(action_dict)  # dictionary with action for each agent
    # print(single_env.action_space.sample())  # return a number between 0-18 as action
    agents_dict = _init_agents(agent_names, shared_weights)

    for agent_name, agent in agents_dict.items():
        action_dict[agent_name] = agent.choose_action(observation[agent_name])
        print(observation[agent_name].shape)

    observation_, reward, done, info = single_env.step(action_dict)
    for agent_name, agent in agents_dict.items():
        print(observation[agent_name].shape)
        print(observation_[agent_name].shape)
        print(reward[agent_name])
        print(done['__all__'])

    n_games = 150
    scores = []
    eps_history = []
    best_reward = 0
    for i in range(n_games):
        isdone = False
        score = 0
        observation = single_env.reset()
        while not isdone:
            for agent_name in agents_dict.keys():
                action_dict[agent_name] = agents_dict[agent_name].choose_action(observation[agent_name])
            # action = agent.choose_action(observation)
            observation_, reward, done, info = single_env.step(action_dict)
            # print(done['__all__'])
            isdone = done['__all__']
            for agent_name in agents_dict.keys():
                # action_dict[agent_name] = agent.choose_action(observation[agent_name])
                score += reward[agent_name]
                agents_dict[agent_name].store_transition(observation[agent_name], action_dict[agent_name],
                                                         reward[agent_name], observation_[agent_name], done['__all__'])

            observation = observation_

        for agent_name in agents_dict.keys():
            agents_dict[agent_name].learn()

        eps_history.append(agents_dict['agent_0'].epsilon)
        print(score / len(agent_names), agents_dict['agent_0'].epsilon)
        scores.append(score / len(agent_names))

    filename = './dqn_grf.png'

    x = [i + 1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)
