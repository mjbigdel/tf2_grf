import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

# Model
class Network:
    def __init__(self, config, agent_ids):
        # super().__init__()
        self.config = config
        self.agent_ids = agent_ids

    def init_base_model(self):
        if self.config.network == 'cnn':
            return cnn(self.config)
        if self.config.network == 'mlp':
            return mlp(self.config)
        if self.config.network == 'impala_cnn':
            return impala_cnn(self.config)

    def build_models_and_agent_heads(self, name):
        """
        :return: list of heads for agents
            - gets tensorflow model and adds heads for each agent
        """
        base_model = self.init_base_model()
        head_inputs = base_model.layers[-1].output
        heads = []
        for agent_id in self.agent_ids:
            name += self.config.network + f'_agent_{str(agent_id)}'
            if self.config.noisy_layers:
                outputs_a = noisy_dense(self.config.num_actions)(head_inputs)
            else:
                outputs_a = dense(self.config.num_actions, name, k_init=tf.keras.initializers.Orthogonal(1.0),
                                  b_init=tf.keras.initializers.Constant(0.0), act=None)(head_inputs)

            head = tf.keras.Model(inputs=base_model.inputs, outputs=outputs_a, name=name)
            heads.append(head)
        return heads

    def build_models_and_dueling_agent_heads(self, name):
        """
        :return: list of heads for agents
            - gets tensorflow model and adds heads for each agent
        """
        base_model = self.init_base_model()
        head_inputs = base_model.layers[-1].output
        heads = []
        for agent_id in self.agent_ids:
            name += self.config.network + f'_agent_{str(agent_id)}'
            with tf.name_scope(f'action_value_{agent_id}'):
                if self.config.noisy_layers:
                    action_head_a = noisy_dense(self.config.num_actions)(head_inputs)
                else:
                    action_head_a = dense(self.config.num_actions, 'action_' + name, k_init=tf.keras.initializers.Orthogonal(1.0),
                                          b_init=tf.keras.initializers.Constant(0.0), act=None)(head_inputs)

            with tf.name_scope(f'state_value_{agent_id}'):
                if self.config.noisy_layers:
                    state_head_a = noisy_dense(1)(head_inputs)
                else:
                    state_head_a = dense(1, 'state_' + name, k_init=tf.keras.initializers.Orthogonal(1.0),
                                         b_init=tf.keras.initializers.Constant(0.0), act=None)(head_inputs)

            action_scores_mean = tf.reduce_mean(action_head_a, 1)
            action_scores_centered = action_head_a - tf.expand_dims(action_scores_mean, 1)
            head = state_head_a + action_scores_centered

            head = tf.keras.Model(inputs=base_model.inputs, outputs=head)
            heads.append(head)

        return heads

    def build_models_and_agent_heads_dist(self, name):
        """
        :return: list of heads for agents
            - gets tensorflow model and adds heads for each agent
        """
        base_model = self.init_base_model()
        head_inputs = base_model.layers[-1].output
        heads = []
        for agent_id in self.agent_ids:
            name += self.config.network + f'_agent_{str(agent_id)}'
            if self.config.noisy_layers:
                outputs_a = noisy_dense(self.config.num_actions * self.config.atoms)(head_inputs)
                outputs_a = tf.keras.layers.Activation(tf.nn.relu, name=f'{name}_act_noisyLayer')(outputs_a)
            else:
                outputs_a = dense(self.config.num_actions * self.config.atoms, name, k_init=tf.keras.initializers.Orthogonal(1.0),
                                  b_init=tf.keras.initializers.Constant(0.0), act='relu')(head_inputs)

            outputs_a = tf.keras.layers.Reshape([self.config.num_actions, self.config.atoms], name=f'reshape_{name}')(outputs_a)
            head = tf.keras.Model(inputs=base_model.inputs, outputs=outputs_a, name=name)
            heads.append(head)
        return heads

    def build_models(self, name):
        if self.config.distributionalRL:
            return self.build_models_and_agent_heads_dist(name)

        if self.config.dueling:
            return self.build_models_and_dueling_agent_heads(name)
        else:
            return self.build_models_and_agent_heads(name)


def conv(num_ch, ks, st, pad, name, k_init, b_init):
    return tf.keras.layers.Conv2D(filters=num_ch, kernel_size=ks, strides=st, padding=pad, name=f'conv2_{name}',
                                  kernel_initializer=k_init, bias_initializer=b_init)


def dense(units, name, k_init, b_init, act=tf.keras.activations.relu):
    return tf.keras.layers.Dense(units=units, activation=act, kernel_initializer=k_init,
                                 bias_initializer=b_init, name=name)


def noisy_dense(units, std_init=0.5):
    # return NoisyDense(units, std_init=0.5)
    return tfa.layers.NoisyDense(int(units), std_init)


# Factorized Gaussian Noise Layer
# Reference from https://github.com/Kaixhin/Rainbow/blob/master/model.py
class NoisyDense(tf.keras.layers.Layer):
    def __init__(self, units, std_init=0.5):
        super().__init__()
        self.units = units
        self.std_init = std_init

    def call(self, inputs):
        input_dim = inputs.shape[-1]
        # print(f'input_dim is {input_dim}')
        self.reset_noise(input_dim)
        mu_range = 1 / np.sqrt(input_dim)
        mu_initializer = tf.initializers.RandomUniform(-mu_range, mu_range)
        sigma_initializer = tf.initializers.Constant(self.std_init / np.sqrt(self.units))
        self.weight_mu = self.add_weight(
            shape=(input_dim, self.units), initializer=mu_initializer, trainable=True, dtype='float32'
        )

        self.weight_sigma = self.add_weight(
            shape=(input_dim, self.units), initializer=sigma_initializer, trainable=True, dtype='float32'
        )

        self.bias_mu = self.add_weight(
            shape=(self.units,), initializer=mu_initializer, trainable=True, dtype='float32'
        )

        self.bias_sigma = self.add_weight(
            shape=(self.units,), initializer=sigma_initializer, trainable=True, dtype='float32'
        )

        # with tf.init_scope():
        #     self.weight_mu = tf.Variable(initial_value=mu_initializer(shape=(input_dim, self.units), dtype='float32'),
        #                                  trainable=True)
        #
        #     self.weight_sigma = tf.Variable(initial_value=sigma_initializer(shape=(input_dim, self.units), dtype='float32'),
        #                                     trainable=True)
        #
        #     self.bias_mu = tf.Variable(initial_value=mu_initializer(shape=(self.units,), dtype='float32'),
        #                                trainable=True)
        #
        #     self.bias_sigma = tf.Variable(initial_value=sigma_initializer(shape=(self.units,), dtype='float32'),
        #                               trainable=True)
        # output = tf.tensordot(inputs, self.kernel, 1)
        # tf.nn.bias_add(output, self.bias)
        # return output
        self.kernel = self.weight_mu + self.weight_sigma * self.weights_eps
        self.bias = self.bias_mu + self.bias_sigma * self.bias_eps
        return tf.matmul(inputs, self.kernel) + self.bias

    def _scale_noise(self, dim):
        noise = tf.random.normal([dim])
        return tf.sign(noise) * tf.sqrt(tf.abs(noise))

    def reset_noise(self, input_shape):
        eps_in = self._scale_noise(input_shape)
        eps_out = self._scale_noise(self.units)
        self.weights_eps = tf.multiply(tf.expand_dims(eps_in, 1), eps_out)
        self.bias_eps = eps_out

def cnn(config):
    # basic variables
    conv_layers = config.conv_layers
    kernel_init = tf.keras.initializers.Orthogonal(gain=1.0)
    bias_init = tf.keras.initializers.Constant(0.0)

    # inputs
    inputs_obs = tf.keras.layers.Input(shape=config.obs_shape)

    # neural network
    conv_out = inputs_obs
    conv_out = tf.cast(conv_out, dtype=tf.float32) / 255.
    for i, layer in enumerate(conv_layers):
        conv_out = conv(layer[0], layer[1], layer[2], 'same', f'conv_{i}', kernel_init, bias_init)(conv_out)
        conv_out = tf.keras.layers.Activation(tf.nn.relu)(conv_out)

    conv_out = tf.keras.layers.Flatten(name='flatten')(conv_out)
    if config.noisy_layers:
        conv_out = noisy_dense(config.fc1_dims)(conv_out)
        conv_out = tf.keras.layers.Activation(tf.nn.relu, name=f'act_noisyLayer')(conv_out)
    else:
        conv_out = dense(config.fc1_dims, 'dense1', kernel_init, bias_init, tf.nn.relu)(conv_out)
    return tf.keras.Model(inputs=inputs_obs, outputs=conv_out, name=config.network)


def mlp(config):
    # basic variables
    kernel_init = tf.keras.initializers.Orthogonal(gain=1.0)
    bias_init = tf.keras.initializers.Constant(0.0)

    # inputs
    inputs_obs = tf.keras.layers.Input(config.obs_shape, name='Input_obs')

    # neural network
    outputs = tf.keras.layers.Flatten(name='flatten')(inputs_obs)

    if config.noisy_layers:
        outputs = noisy_dense(config.fc1_dims)(outputs)
        outputs = tf.keras.layers.Activation(tf.nn.relu, name=f'act_noisyLayer')(outputs)
    else:
        outputs = dense(config.fc1_dims, 'dense1', kernel_init, bias_init, tf.nn.relu)(outputs)

    return tf.keras.Model(inputs=inputs_obs, outputs=outputs, name=config.network)


def impala_cnn(config):
    """

    :param config: config object containing all parameters and hyper-parameters
        resnet style with config.impala_layers inspired from IMPALA paper
    :return: Tensorflow Keras Model
    """
    # basic variables
    impala_layers = config.impala_layers
    kernel_init = tf.keras.initializers.Orthogonal(gain=1.0)
    bias_init = tf.keras.initializers.Constant(0.0)

    # inputs
    inputs_obs = tf.keras.layers.Input(shape=config.obs_shape)

    # neural network
    conv_out = inputs_obs
    conv_out = tf.cast(conv_out, dtype=tf.float32) / 255.

    for i, (num_ch, num_blocks) in enumerate(impala_layers):
        conv_out = conv(num_ch, 3, 1, 'same', f'conv_{i}', kernel_init, bias_init)(conv_out)
        conv_out = tf.keras.layers.MaxPooling2D((3, 3), padding='same', strides=2, name=f'maxP_{i}')(conv_out)

        for j in range(num_blocks):
            with tf.name_scope('residual_%d_%d' % (i, j)):
                block_input = conv_out
                conv_out = tf.keras.layers.Activation(tf.nn.relu, name=f'act_{i}_{j}_1')(conv_out)
                conv_out = conv(num_ch, 3, 1, 'same', f'conv_{i}_{j}_1', kernel_init, bias_init)(conv_out)
                conv_out = conv(num_ch, 3, 1, 'same', f'conv_{i}_{j}_2', kernel_init, bias_init)(conv_out)

                conv_out += block_input

    conv_out = tf.keras.layers.Activation(tf.nn.relu)(conv_out)
    conv_out = tf.keras.layers.BatchNormalization()(conv_out)
    conv_out = tf.keras.layers.Flatten()(conv_out)
    if config.noisy_layers:
        conv_out = noisy_dense(config.fc1_dims)(conv_out)
        conv_out = tf.keras.layers.Activation(tf.nn.relu, name=f'act_noisyLayer')(conv_out)
    else:
        conv_out = dense(config.fc1_dims, 'dense1', kernel_init, bias_init, tf.nn.relu)(conv_out)

    return tf.keras.Model(inputs=inputs_obs, outputs=conv_out, name=config.network)


