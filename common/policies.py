
import tensorflow as tf
from common.distributions import make_pdtype
from a2c_1.utils import ortho_init, fc, fc_build
import tensorflow.keras.backend as K

class PolicyWithValue(tf.Module):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, ac_space, policy_network, value_network=None, estimate_q=False):
        """
        Parameters:
        ----------
        ac_space        action space

        policy_network  keras network for policy

        value_network   keras network for value

        estimate_q      q value or v value

        """

        self.policy_network = policy_network
        self.value_network = value_network or policy_network
        self.estimate_q = estimate_q
        self.initial_state = None
        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(self.policy_network.output_shape, ac_space, init_scale=0.01)

        if estimate_q:
            self.value_fc = fc_build(self.value_network.output_shape, 'q', ac_space.n)
        else:
            self.value_fc = fc_build(self.value_network.output_shape, 'vf', 1)

        # # to get just dense size and avoid batch size
        # print(f'self.value_network.output_shape for agent_0 {self.value_network.output_shape[-1]}')
        # value_model_inputes = tf.keras.layers.Input(self.value_network.output_shape[-1])  # agent 0 output
        #
        # if estimate_q:
        #     # value_fc = fc(scope='q', nh=ac_space.n)(policy_network.output)
        #     value_fc = tf.keras.layers.Dense(units=ac_space.n, kernel_initializer=ortho_init(init_scale),
        #                                      bias_initializer=tf.keras.initializers.Constant(init_bias),
        #                                      name=f'q')(value_model_inputes)
        # else:
        #     # value_fc = fc(scope='vf', nh=1)(policy_network.output)
        #     value_fc = tf.keras.layers.Dense(units=1, kernel_initializer=ortho_init(init_scale),
        #                                      bias_initializer=tf.keras.initializers.Constant(init_bias),
        #                                      name=f'vf')(value_model_inputes)

        # self.value_model = tf.keras.Model(inputs=value_model_inputes, outputs=value_fc, name='Value_Network')
        # self.value_model.summary()
        # tf.keras.utils.plot_model(self.value_model, to_file='./value_model.png')

        self.value_network.summary()
        self.policy_network.summary()
        tf.keras.utils.plot_model(self.policy_network, to_file='./policy_network.png')

    @tf.function
    def step(self, observation):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     batched observation data

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        latent = self.policy_network(observation)
        # print(f'latent is {latent}')
        pd, pi = self.pdtype.pdfromlatent(latent)
        # print(f'pd, pi  is {pd, pi }')
        action = pd.sample()
        # print(f'action is {action}')
        neglogp = pd.neglogp(action)
        value_latent = self.value_network(observation)
        vf = tf.squeeze(self.value_fc(value_latent), axis=1)
        # print(f'vf is {vf}')
        return action, vf, None, neglogp

    @tf.function
    def value(self, observation):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        Returns:
        -------
        value estimate
        """
        value_latent = self.value_network(observation)
        result = tf.squeeze(self.value_fc(value_latent), axis=1)
        return result


class MAPolicyWithValue(tf.Module):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, agent_ids, num_agents, ac_space, policy_network,
                 value_network=None, estimate_q=False):
        """
        Parameters:
        ----------
        ac_space        action space

        policy_network  keras network for policy

        value_network   keras network for value

        estimate_q      q value or v value

        """
        self.num_agents = num_agents
        self.agent_ids = agent_ids

        self.policy_network = policy_network
        self.value_network = value_network or policy_network
        self.estimate_q = estimate_q
        self.initial_state = None
        self.ac_space = ac_space

        print(f'self.policy_network.output_shape {self.policy_network.output_shape}')
        print(f'self.value_network.output_shape {self.value_network.output_shape}')

        self.pdtypes = self._build_actor_head()
        self.critics_fc = self._build_critic_head()

        print(f'self.pdtype is {self.pdtypes}')
        print(f'self.critics_fc is {self.critics_fc}')

        self.policy_network.summary()
        tf.keras.utils.plot_model(self.policy_network, to_file='./policy_network.png')

    @tf.function
    def _build_actor_head(self):
        pdtypes = []
        input_shape = self.policy_network.output_shape
        for a in self.agent_ids:
            pdtypes.append(make_pdtype(input_shape, self.ac_space, init_scale=0.01))
        return pdtypes

    @tf.function
    def _build_critic_head(self):
        input_shape = self.value_network.output_shape
        name = 'vf'
        critics_fc = []
        for a in self.agent_ids:
            name += '_agent_' + str(a)
            critics_fc.append(fc_build(input_shape, name, 1))
        return critics_fc

    @tf.function
    def step(self, observation):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     batched observation data

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """
        policy_latent = self.policy_network(observation)
        value_latent = self.value_network(observation)


        vfs = []
        actions = []
        neglogps = []
        for a in self.agent_ids:
            pd, pi = self.pdtypes[a].pdfromlatent(tf.expand_dims(policy_latent[a], 0))
            action = pd.sample()
            actions.append(action.numpy()[0])
            neglogp = pd.neglogp(action)
            neglogps.append(neglogp)
            vf = tf.squeeze(self.critics_fc[a](tf.expand_dims(value_latent[a], 0)), axis=1)
            vfs.append(vf.numpy()[0])

        return actions, vfs, None, neglogps

    @tf.function
    def value(self, observation):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        agent_name      name of agent in which observation belongs.

        Returns:
        -------
        value estimate
        """

        value_latent = self.value_network(observation)
        results = []
        for a in self.agent_ids:
            if value_latent[a].ndim == 2:
                result = tf.squeeze(self.critics_fc[a](value_latent[a]), axis=1)
                results.append(result.numpy()[0])
            else:
                result = tf.squeeze(self.critics_fc[a](tf.expand_dims(value_latent[a], 0)), axis=1)
                results.append(result.numpy()[0])
        print(f'results is {results}')
        return results


