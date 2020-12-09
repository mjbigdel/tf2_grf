import numpy as np
from abc import ABC, abstractmethod

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs = env.reset()
        # print(self.obs)
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    @abstractmethod
    def run(self):
        raise NotImplementedError


class MAAbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        # self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
        # self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs = env.reset()  # each obs is a list of dictionaru
        # print(self.obs)
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
        # self.agent_names = [agent_name for agent_name in self.obs[0].keys()]
        # print(self.agent_names)

    @abstractmethod
    def run(self):
        raise NotImplementedError

