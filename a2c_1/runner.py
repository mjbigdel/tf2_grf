import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from a2c_1.utils import discount_with_dones
from common.runners import AbstractEnvRunner
import matplotlib.pyplot as plt

class Runner(AbstractEnvRunner):
    """
    We use this class to generate batches of experiences

    __init__:
    - Initialize the runner

    run():
    - Make a mini batch of experiences
    """
    def __init__(self, env, model, nsteps=5, gamma=0.99):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma

    def run(self):
        # We initialize the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        epinfos = []

        for _ in range(self.nsteps):
            # Given observations, take action and value (V(s))
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            obs = tf.constant(self.obs)
            actions, values, self.states, _ = self.model.step(obs)
            # print(f'actions and values is {actions} ==== {values}')
            # print(f'dones is {self.dones}')
            actions = actions #  K.eval() returns tensor value
            # Append the experiences
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            # values = values._numpy()  # not working
            mb_values.append(values)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            self.obs1, rewards, self.dones, infos = self.env.step(actions)
            # print(f'self.obs.shape {self.obs1.shape}')
            # plt.imshow(self.obs1[0,:,:,0])
            # plt.show()
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
            self.obs = self.obs1

        mb_dones.append(self.dones)

        # Batch of steps to batch of rollouts
        mb_obs = sf01(np.asarray(mb_obs, dtype=self.obs.dtype))
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = sf01(np.asarray(mb_actions, dtype=actions.dtype))
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]

        if self.gamma > 0.0:
            # Discount/bootstrap off value fn
            last_values = self.model.value(tf.constant(self.obs)).tolist()
            for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)

                mb_rewards[n] = rewards


        mb_rewards = mb_rewards.flatten()
        # print(f'mb_rewards is {mb_rewards}')
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, epinfos

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
