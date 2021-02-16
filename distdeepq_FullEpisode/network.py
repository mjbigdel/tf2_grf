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
        if self.config.network == 'cnn_rnn':
            return cnn_rnn(self.config)
        if self.config.network == 'mlp_rnn':
            return mlp_rnn(self.config)
        if self.config.network == 'impala_cnn_rnn':
            return impala_cnn_rnn(self.config)

    def build_models_and_agent_heads(self, name):
        """
        :return: list of heads for agents
            - gets tensorflow model and adds heads for each agent
        """
        base_model = self.init_base_model()
        head_inputs = base_model.layers[-1].output
        if self.config.is_recurrent:
            head_inputs = head_inputs[0]  # output of LSTM
            head_states = head_inputs[1]  # states of LSTM
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

    def build_models_and_agent_heads_dist(self, name):
        """
        :return: list of heads for agents
            - gets tensorflow model and adds heads for each agent
        """
        base_model = self.init_base_model()
        head_inputs = base_model.layers[-1].output
        if self.config.is_recurrent:
            head_inputs = head_inputs[0]  # output of LSTM
            head_states = head_inputs[1]  # states of LSTM
        # print(f'head_inputs is {head_inputs}')
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

    def build_models_and_dueling_agent_heads(self, name):
        """
        :return: list of heads for agents
            - gets tensorflow model and adds heads for each agent
        """
        base_model = self.init_base_model()
        head_inputs = base_model.layers[-1].output
        if self.config.is_recurrent:
            head_inputs = head_inputs[0]  # output of LSTM
            head_states = head_inputs[1]  # states of LSTM
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

    def build_models_and_dueling_agent_heads_dist(self, name):
        """
        :return: list of heads for agents
            - gets tensorflow model and adds heads for each agent
        """
        base_model = self.init_base_model()
        head_inputs = base_model.layers[-1].output
        if self.config.is_recurrent:
            head_inputs = head_inputs[0]  # output of LSTM
            head_states = head_inputs[1]  # states of LSTM
        # print(f'head_inputs is {head_inputs}')
        heads = []

        for agent_id in self.agent_ids:
            name += self.config.network + f'_agent_{str(agent_id)}'
            with tf.name_scope(f'action_value_{agent_id}'):
                if self.config.noisy_layers:
                    action_head_a = noisy_dense(self.config.num_actions * self.config.atoms)(head_inputs)
                    action_head_a = tf.keras.layers.Activation(tf.nn.relu, name=f'{name}_act_noisyLayer')(action_head_a)
                else:
                    action_head_a = dense(self.config.num_actions * self.config.atoms, name, k_init=tf.keras.initializers.Orthogonal(1.0),
                                          b_init=tf.keras.initializers.Constant(0.0), act='relu')(head_inputs)

            with tf.name_scope(f'state_value_{agent_id}'):
                if self.config.noisy_layers:
                    state_head_a = noisy_dense(1)(head_inputs)
                    state_head_a = tf.keras.layers.Activation(tf.nn.relu, name=f'state_{name}_act_noisyLayer')(state_head_a)
                else:
                    state_head_a = dense(1, 'state_' + name, k_init=tf.keras.initializers.Orthogonal(1.0),
                                         b_init=tf.keras.initializers.Constant(0.0), act='relu')(head_inputs)

            action_scores_mean = tf.reduce_mean(action_head_a, 1)
            action_scores_centered = action_head_a - tf.expand_dims(action_scores_mean, 1)
            duel_out = state_head_a + action_scores_centered
            duel_out = tf.keras.layers.Reshape([self.config.num_actions, self.config.atoms],
                                               name=f'reshape_{name}')(duel_out)

            head = tf.keras.Model(inputs=base_model.inputs, outputs=duel_out)
            heads.append(head)

        return heads

    def build_models(self, name):
        if self.config.distributionalRL:
            if self.config.dueling:
                return self.build_models_and_dueling_agent_heads_dist(name)
            else:
                return self.build_models_and_agent_heads_dist(name)
        else:
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


def noisy_dense(units, scope='NoisyDense', std_init=0.5):
    return NoisyDense(units, scope=scope, std_init=std_init)
    # return tfa.layers.NoisyDense(int(units), std_init)


# Factorized Gaussian Noise Layer
# Reference from https://github.com/Kaixhin/Rainbow/blob/master/model.py
class NoisyDense(tf.keras.layers.Layer):
    def __init__(self, units, scope, std_init=0.5):
        super(NoisyDense, self).__init__()
        self.units = units
        self.std_init = std_init
        self.scope = scope

    def build(self, input_shape):
        input_dim = input_shape[-1]
        # print(f'input_dim is {input_dim}')
        self.reset_noise(input_dim)
        mu_range = 1 / np.sqrt(input_dim)
        mu_initializer = tf.initializers.RandomUniform(-mu_range, mu_range)
        sigma_initializer = tf.initializers.Constant(self.std_init / np.sqrt(self.units))

        with tf.name_scope(self.scope):
            self.weight_mu = self.add_weight(shape=(input_dim, self.units), initializer=mu_initializer,
                                             trainable=True, dtype='float32', name="w_mu")

            self.weight_sigma = self.add_weight(shape=(input_dim, self.units), initializer=sigma_initializer,
                                                trainable=True, dtype='float32', name="w_sig")

            self.bias_mu = self.add_weight(shape=(self.units,), initializer=mu_initializer,
                                           trainable=True, dtype='float32', name="b_mu")

            self.bias_sigma = self.add_weight(shape=(self.units,), initializer=sigma_initializer,
                                              trainable=True, dtype='float32', name="b_sig")

    def call(self, inputs):
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

    # def compute_output_shape(self, input_shape):
    #
    #     input_shape = tensor_shape.TensorShape(input_shape).as_list()
    #     return tensor_shape.TensorShape([input_shape[0]] + input_shape[2:])

    def compute_output_shape(self, input_shape):
        from tensorflow.python.framework import tensor_shape
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)


# initializers
# """
DEFAULT_SCALE = np.sqrt(2)
DEFAULT_MODE = 'fan_in'


def ortho_init(scale=DEFAULT_SCALE, mode=None):
    def _ortho_init(shape, dtype, partition_info=None):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2: # fc: in, out
            flat_shape = shape
        elif (len(shape) == 3) or (len(shape) == 4):  # 1d/2dcnn: (in_h), in_w, in_c, out
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        a = np.random.standard_normal(flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q).astype(np.float32)
    return _ortho_init


def norm_init(scale=DEFAULT_SCALE, mode=DEFAULT_MODE):
    def _norm_init(shape, dtype, partition_info=None):
        shape = tuple(shape)
        if len(shape) == 2:
            n_in = shape[0]
        elif (len(shape) == 3) or (len(shape) == 4):
            n_in = np.prod(shape[:-1])
        a = np.random.standard_normal(shape)
        if mode == 'fan_in':
            n = n_in
        elif mode == 'fan_out':
            n = shape[-1]
        elif mode == 'fan_avg':
            n = 0.5 * (n_in + shape[-1])
        return (scale * a / np.sqrt(n)).astype(np.float32)


DEFAULT_METHOD = ortho_init


class LSTM_M(tf.keras.layers.Layer):
    def __init__(self, units=32, scope='LSTM_M', init_scale=DEFAULT_SCALE,
                 init_mode=DEFAULT_MODE, init_method=DEFAULT_METHOD, return_sequence=False):
        super(LSTM_M, self).__init__()
        # print('init ----- ')
        self.units = units
        self.scope = scope
        self.init_scale = init_scale
        self.init_mode = init_mode
        self.init_method = init_method
        self.return_sequence = return_sequence

    def build(self, input_shape):
        # print(f'input_shape is {input_shape[0][-1]}')

        n_in = input_shape[0][-1]
        # self.n_out = input_shape[0][0] // 2
        self.n_out = self.units
        self.states = np.zeros(self.n_out * 2, dtype=np.float32)
        # print('build ----- ')
        # print(f'n_in is {n_in}')
        # print(f'self.n_out is {self.n_out}')

        with tf.name_scope(self.scope):
            self.wx = self.add_weight(shape=(n_in, self.n_out * 4), initializer='uniform', name="wx")
            self.wh = self.add_weight(shape=(self.n_out, self.n_out * 4), initializer='uniform', name="wh")
            self.b = self.add_weight(shape=(self.n_out * 4), initializer=tf.constant_initializer(0.0), name="b")

    def call(self, inputs, **kwargs):
        # print('call ----- ')
        xs = self.batch_to_seq(inputs[0])
        # print(f'xs {xs}')
        # need dones to reset states
        dones = self.batch_to_seq(inputs[1])
        # print(f'dones {dones}')

        s = tf.expand_dims(self.states, 0)
        # print(f's {s}')
        c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
        # print(f'c {c}')
        # print(f'h {h}')
        for ind, (x, done) in enumerate(zip(xs, dones)):
            c = c * (1 - done)
            h = h * (1 - done)
            z = tf.matmul(x, self.wx)
            z += tf.matmul(h, self.wh)
            z += self.b
            # print(f'z {z}')
            i, f, o, u = tf.split(axis=2, num_or_size_splits=4, value=z)
            # print(f'i {i}')
            # print(f'f {f}')
            # print(f'o {o}')
            # print(f'u {u}')
            i = tf.nn.sigmoid(i)
            f = tf.nn.sigmoid(f)
            o = tf.nn.sigmoid(o)
            u = tf.tanh(u)
            # print(f'c {c}')
            c = f * c + i * u
            h = o * tf.tanh(c)
            xs[ind] = h
        s = tf.concat(axis=2, values=[c, h])
        # print(f's {s.shape}')
        # print(f'xs {xs}')
        xs = self.seq_to_batch(xs)
        s = tf.squeeze(s)
        # print(f's {s}')
        # print(f'xs {xs}')
        if self.return_sequence:
            return xs, s
        else:
            return xs[:, -1], s

    def batch_to_seq(self, x):
        n_step = x.shape[1]
        if len(x.shape) == 1:
            x = tf.expand_dims(x, -1)
        return tf.split(axis=1, num_or_size_splits=n_step, value=x)

    def seq_to_batch(self, x):
        return tf.concat(x, axis=1)


def cnn(config):
    # basic variables
    conv_layers = config.conv_layers
    kernel_init = tf.keras.initializers.Orthogonal(gain=1.0)
    bias_init = tf.keras.initializers.Constant(0.0)

    # inputs
    inputs_obs = tf.keras.layers.Input(shape=config.obs_shape)

    # neural network
    if config.normalize_inputs:
        conv_out = inputs_obs
        conv_out = tf.keras.layers.Lambda(lambda x: (2 * x - 255) / 255.0, )(conv_out)
        # conv_out = tf.cast(conv_out, dtype=tf.float32) / 255.
    else:
        conv_out = inputs_obs

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


def cnn_rnn(config):
    # basic variables
    conv_layers = config.conv_layers
    kernel_init = tf.keras.initializers.Orthogonal(gain=1.0)
    bias_init = tf.keras.initializers.Constant(0.0)
    TdL = tf.keras.layers.TimeDistributed

    # inputs
    inputs_obs = tf.keras.layers.Input(shape=(config.n_steps, *config.obs_shape))
    done_Input = tf.keras.Input(shape=(config.n_steps, 1))
    inputs_fps = tf.keras.Input(shape=(config.n_steps, config.fp_shape))

    # neural network
    if config.normalize_inputs:
        conv_out = inputs_obs
        conv_out = tf.keras.layers.Lambda(lambda x: (2 * x - 255) / 255.0, )(conv_out)
        # conv_out = tf.cast(conv_out, dtype=tf.float32) / 255.
    else:
        conv_out = inputs_obs

    for i, layer in enumerate(conv_layers):
        conv_out = TdL(conv(layer[0], layer[1], layer[2], 'same', f'conv_{i}', kernel_init, bias_init))(conv_out)
        conv_out = tf.keras.layers.Activation(tf.nn.relu)(conv_out)

    conv_out = TdL(tf.keras.layers.Flatten(name='flatten'))(conv_out)

    if config.noisy_layers:
        conv_out = TdL(noisy_dense(config.fc1_dims, scope='noisy_dense1'))(conv_out)
        conv_out = tf.keras.layers.Activation(tf.nn.relu, name=f'act_noisyLayer')(conv_out)

        fps_dense = TdL(noisy_dense(config.fps_dense_dim, scope='noisy_fps_dense'))(inputs_fps)
        fps_dense = tf.keras.layers.Activation(tf.nn.relu, name=f'act_noisy_fps_dense')(fps_dense)
    else:
        conv_out = TdL(dense(config.fc1_dims, 'dense1', kernel_init, bias_init, tf.nn.relu))(conv_out)
        fps_dense = TdL(dense(config.fps_dense_dim, 'noisy_fps_dense', kernel_init, bias_init, tf.nn.relu))(inputs_fps)

    outputs = tf.keras.layers.concatenate([conv_out, fps_dense])

    # print(f'conv_out, {conv_out}')
    # print(f'done_Input, {done_Input}')
    lstm_layer = LSTM_M(config.rnn_dim, return_sequence=False)([outputs, done_Input])

    return tf.keras.Model(inputs=[inputs_obs, done_Input, inputs_fps], outputs=lstm_layer, name=config.network)


def mlp(config):
    # basic variables
    kernel_init = tf.keras.initializers.Orthogonal(gain=1.0)
    bias_init = tf.keras.initializers.Constant(0.0)

    # inputs
    inputs_obs = tf.keras.layers.Input(config.obs_shape, name='Input_obs')

    if config.normalize_inputs:
        # inputs_obs = tf.cast(inputs_obs, dtype=tf.float32) / 255.
        inputs_obs = tf.keras.layers.Lambda(lambda x: (2 * x - 255) / 255.0, )(inputs_obs)

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
    if config.normalize_inputs:
        conv_out = inputs_obs
        conv_out = tf.keras.layers.Lambda(lambda x: (2 * x - 255) / 255.0, )(conv_out)
        # conv_out = tf.cast(conv_out, dtype=tf.float32) / 255.
    else:
        conv_out = inputs_obs

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


def impala_cnn_rnn(config):
    """

    :param config: config object containing all parameters and hyper-parameters
        resnet style with config.impala_layers inspired from IMPALA paper
    :return: Tensorflow Keras Model
    """
    # basic variables
    impala_layers = config.impala_layers
    kernel_init = tf.keras.initializers.Orthogonal(gain=1.0)
    bias_init = tf.keras.initializers.Constant(0.0)
    TdL = tf.keras.layers.TimeDistributed

    # inputs
    inputs_obs = tf.keras.layers.Input(shape=(config.n_steps, *config.obs_shape))
    done_Input = tf.keras.Input(shape=(config.n_steps, 1))
    inputs_fps = tf.keras.Input(shape=(config.n_steps, config.fp_shape))

    # neural network
    if config.normalize_inputs:
        conv_out = inputs_obs
        conv_out = tf.keras.layers.Lambda(lambda x: (2 * x - 255) / 255.0,)(conv_out)
        # conv_out = tf.cast(conv_out, dtype=tf.float32) / 255.
    else:
        conv_out = inputs_obs

    for i, (num_ch, num_blocks) in enumerate(impala_layers):
        conv_out = TdL(conv(num_ch, 3, 1, 'same', f'conv_{i}', kernel_init, bias_init))(conv_out)
        conv_out = TdL(tf.keras.layers.MaxPooling2D((3, 3), padding='same', strides=2, name=f'maxP_{i}'))(conv_out)

        for j in range(num_blocks):
            with tf.name_scope('residual_%d_%d' % (i, j)):
                block_input = conv_out
                conv_out = TdL(tf.keras.layers.Activation(tf.nn.relu, name=f'act_{i}_{j}_1'))(conv_out)
                conv_out = TdL(conv(num_ch, 3, 1, 'same', f'conv_{i}_{j}_1', kernel_init, bias_init))(conv_out)
                conv_out = TdL(conv(num_ch, 3, 1, 'same', f'conv_{i}_{j}_2', kernel_init, bias_init))(conv_out)

                conv_out += block_input

    conv_out = tf.keras.layers.Activation(tf.nn.relu)(conv_out)
    conv_out = tf.keras.layers.BatchNormalization()(conv_out)
    conv_out = TdL(tf.keras.layers.Flatten())(conv_out)

    if config.noisy_layers:
        conv_out = TdL(noisy_dense(config.fc1_dims))(conv_out)
        conv_out = tf.keras.layers.Activation(tf.nn.relu, name=f'act_noisyLayer')(conv_out)

        fps_dense = TdL(noisy_dense(config.fps_dense_dim, scope='noisy_fps_dense'))(inputs_fps)
        fps_dense = tf.keras.layers.Activation(tf.nn.relu, name=f'act_noisy_fps_dense')(fps_dense)
    else:
        conv_out = TdL(dense(config.fc1_dims, 'dense1', kernel_init, bias_init, tf.nn.relu))(conv_out)
        fps_dense = TdL(dense(config.fps_dense_dim, 'noisy_fps_dense', kernel_init, bias_init, tf.nn.relu))(inputs_fps)

    outputs = tf.keras.layers.concatenate([conv_out, fps_dense])

    lstm_layer = LSTM_M(config.rnn_dim, return_sequence=False)([outputs, done_Input])

    return tf.keras.Model(inputs=[inputs_obs, done_Input, inputs_fps], outputs=lstm_layer, name=config.network)
