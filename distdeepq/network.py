import tensorflow as tf


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

    def build_models_and_agent_heads(self):
        """
        :return: list of heads for agents
            - gets tensorflow model and adds heads for each agent
        """
        base_model = self.init_base_model()
        head_inputs = base_model.layers[-1].output
        heads = []
        for agent_id in self.agent_ids:
            name = self.config.network + f'_agent_{str(agent_id)}'
            outputs_a = dense(self.config.num_actions, name, k_init=tf.keras.initializers.Orthogonal(1.0),
                              b_init=tf.keras.initializers.Constant(0.0), act=None)(head_inputs)

            head = tf.keras.Model(inputs=base_model.inputs, outputs=outputs_a, name=name)
            heads.append(head)
        return heads

    def build_models_and_dueling_agent_heads(self):
        """
        :return: list of heads for agents
            - gets tensorflow model and adds heads for each agent
        """
        base_model = self.init_base_model()
        head_inputs = base_model.layers[-1].output
        heads = []
        for agent_id in self.agent_ids:
            name = self.config.network + f'_agent_{str(agent_id)}'
            with tf.name_scope(f'action_value_{agent_id}'):
                action_head_a = dense(self.config.num_actions, 'action_' + name, k_init=tf.keras.initializers.Orthogonal(1.0),
                                      b_init=tf.keras.initializers.Constant(0.0), act=None)(head_inputs)

            with tf.name_scope(f'state_value_{agent_id}'):
                state_head_a = dense(1, 'state_' + name, k_init=tf.keras.initializers.Orthogonal(1.0),
                                     b_init=tf.keras.initializers.Constant(0.0), act=None)(head_inputs)

            action_scores_mean = tf.reduce_mean(action_head_a, 1)
            action_scores_centered = action_head_a - tf.expand_dims(action_scores_mean, 1)
            head = state_head_a + action_scores_centered

            head = tf.keras.Model(inputs=base_model.inputs, outputs=head)
            heads.append(head)

        return heads

    def build_models_and_agent_heads_dist(self):
        """
        :return: list of heads for agents
            - gets tensorflow model and adds heads for each agent
        """
        base_model = self.init_base_model()
        head_inputs = base_model.layers[-1].output
        heads = []
        for agent_id in self.agent_ids:
            name = self.config.network + f'_agent_{str(agent_id)}'
            outputs_a = dense(self.config.num_actions * self.config.atoms, name, k_init=tf.keras.initializers.Orthogonal(1.0),
                              b_init=tf.keras.initializers.Constant(0.0), act='linear')(head_inputs)

            outputs_a = tf.keras.layers.Reshape([self.config.num_actions, self.config.atoms], name=f'reshape_{name}')(outputs_a)
            head = tf.keras.Model(inputs=base_model.inputs, outputs=outputs_a, name=name)
            heads.append(head)
        return heads

    def build_models(self):
        if self.config.distributionalRL:
            return self.build_models_and_agent_heads_dist()

        if self.config.dueling:
            return self.build_models_and_dueling_agent_heads()
        else:
            return self.build_models_and_agent_heads()


def conv(num_ch, ks, st, pad, name, k_init, b_init):
    return tf.keras.layers.Conv2D(filters=num_ch, kernel_size=ks, strides=st, padding=pad, name=f'conv2_{name}',
                                  kernel_initializer=k_init, bias_initializer=b_init)


def dense(units, name, k_init, b_init, act=tf.keras.activations.relu):
    return tf.keras.layers.Dense(units=units, activation=act, kernel_initializer=k_init,
                                 bias_initializer=b_init, name=name)


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
    conv_out = dense(config.fc1_dims, 'dense1', kernel_init, bias_init, tf.nn.relu)(conv_out)

    return tf.keras.Model(inputs=inputs_obs, outputs=conv_out, name=config.network)




