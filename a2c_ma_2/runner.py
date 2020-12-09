import tensorflow as tf
import numpy as np

from common.runners import MAAbstractEnvRunner, AbstractEnvRunner

from a2c_ma_2.utils import discount_with_dones

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
            # print(f'actions and values is {actions.numpy()} ==== {values.numpy()}')
            # print(f'dones is {self.dones}')
            actions = actions.numpy() #  K.eval() returns tensor value
            # Append the experiences
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            values = values.numpy()  # not working
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
            last_values = self.model.value(tf.constant(self.obs)).numpy().tolist()
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


class MRunner(MAAbstractEnvRunner):
    """
    We use this class to generate batches of experiences

    __init__:
    - Initialize the runner

    run():
    - Make a mini batch of experiences
    """
    def __init__(self, env, model, nsteps=5, gamma=0.99, num_agents=1):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma
        self.num_agents = num_agents
        # self.agent_names = env.agent_names()

    def run(self):
        # We initialize the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        epinfos = []

        for _ in range(self.nsteps):
            # Given observations, take action and value (V(s))
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            # actions_dict = {}
            # values_dict = {}
            actions_list = []
            values_list = []
            # actions_list = [{} for i in range(self.nenv)]
            # values_list = [{} for i in range(self.nenv)]
            # print(f'obs.shape is {np.asarray(self.obs).shape}')
            # print(f'agent obs.shape is {np.asarray(self.obs[:,0,::]).shape}')
            # print(np.asarray([self.obs_dic[i][self.agent_names[0]] for i in range(self.nenv)]).shape)
            for i in range(self.num_agents):
                obs = tf.constant(self.obs[:,i,::])
                # print(obs.shape)
                actions, values, self.states, _ = self.model.step(obs, i)
                # print(f'actions is {actions.numpy()}')
                # print(f'values is {values.numpy()}')

                # for k in range(self.nenv):
                actions_list.append(actions.numpy())
                values_list.append(values.numpy())

            # print(f'actions_dict is {actions_dict}')

            # for key in actions_dict.keys():
            #     for i, val in enumerate(actions_dict[key]):
            #         actions_list[i][key] = val
            #     for i, val in enumerate(values_dict[key]):
            #         values_list[i][key] = val


            # print(f'actions_list.shape is {np.asarray(actions_list).shape}')
            # print(f'values_list.shape is {np.asarray(values_list).shape}')
            actions_list = np.asarray(actions_list).swapaxes(0,1)
            values_list = np.asarray(values_list).swapaxes(0,1)
            # print(f'actions_list.shape is {np.asarray(actions_list).shape}')
            # print(f'values_list.shape is {np.asarray(values_list).shape}')

            # print(f'actions_list is {actions_list}')
            # print(f'values_list is {values_list}')

            # print(values_list)
            # Append the experiences
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions_list)
            # values = values._numpy()  # not working
            mb_values.append(values_list)
            # print(self.dones)
            mb_dones.append(self.dones)

            # print(actions_list.tolist())

            # Take actions in env and look the results
            self.obs1, rewards, self.dones, infos = self.env.step(actions_list.tolist())
            self.dones = [self.dones[k] for k in range(self.nenv)]
            # print(f'self.obs.shape {self.obs1.shape}')
            # plt.imshow(self.obs1[0,:,:,0])
            # plt.show()
            # print(f'rewards is {rewards}')
            for info in infos:
                maybeepinfo = info[0].get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)

            mb_rewards.append(rewards)
            self.obs = self.obs1

        mb_dones.append(self.dones)
        # print('Im here =======================')
        #
        # print(f'mb_actions is {mb_actions}')
        # print(f'mb_rewards is {mb_rewards}')
        # print(f'mb_values is {mb_values}')
        # print(f'mb_dones is {mb_dones}')

        # Batch of steps to batch of rollouts
        mb_obs = sf01(np.asarray(mb_obs, dtype=self.obs.dtype))
        mb_actions = sf01(np.asarray(mb_actions, dtype=actions.numpy().dtype))  #
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)  #
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)  #
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)  #
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]

        # print(f'mb_actions is {mb_actions.shape}')
        # print(f'mb_rewards is {mb_rewards.shape}')
        # print(f'mb_values is {mb_values.shape}')
        # print(f'mb_dones is {mb_dones.shape}')
        # print(f'mb_rewards is {mb_rewards}')
        #
        # print("Im here now ====================================================")

        # print(f'mb_rewards is {mb_rewards}')

        if self.gamma > 0.0:
            # Discount/bootstrap off value fn
            for a in range(self.num_agents):
                last_values = self.model.value(tf.constant(self.obs[:,a,::]), a).numpy().tolist()
                # print(f'last_values is {last_values} for agent {a}')
                for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                    # print(f'rewards[i][{agent_name}] {[rewards[i][agent_name] for i in range(self.nsteps)]}')
                    rewards = [rewards[i][a] for i in range(self.nsteps)]  # rewards.tolist()
                    dones = dones.tolist()
                    # print(f'reward is {rewards}')
                    # print(f'dones is {dones}')
                    # print(f'value is {value}')
                    if dones[-1] == 0:
                        rewards = np.asarray(discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1])
                    else:
                        rewards = np.asarray(discount_with_dones(rewards, dones, self.gamma), dtype=np.float32)

                    # print(f'rewards for {a} {rewards}')
                    for i in range(len(rewards)):
                        mb_rewards[n][i][a] = rewards[i]
                        # print(f'mb_rewards[{n}][{i}][{agent_name}] {mb_rewards[n][i][agent_name]}')


        mb_rewards = sf01(mb_rewards)
        mb_values = sf01(mb_values)
        # print(f'mb_rewards is {mb_rewards}')
        # print(f'mb_actions is {mb_actions.shape}')
        # print(f'mb_rewards is {mb_rewards.shape}')
        # print(f'mb_values is {mb_values.shape}')
        # print(f'mb_obs is {mb_obs.shape}')
        mb_masks = mb_masks.flatten()
        # print(f'mb_rewards is {mb_rewards}')
        # print(f'mb_values is {mb_values}')

        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, epinfos


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
