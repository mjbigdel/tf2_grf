import tensorflow as tf
from numpy.core._multiarray_umath import dtype

tf.config.run_functions_eagerly(True)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from a2c_1.utils import ortho_init, conv
from common.tf2_utils import huber_loss


def build_cnn(input_shape, num_actions, fc1_dims, model_name):
    inputs = tf.keras.layers.Input(shape=input_shape)
    conv1 = tf.keras.layers.Conv2D(filters=96, kernel_size=8, strides=(3, 3),
                                   activation=tf.nn.relu, name='conv1')(inputs)
    conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=(2, 2),
                                   activation=tf.nn.relu, name='conv2')(conv1)
    conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(1, 1),
                                   activation=tf.nn.relu, name='conv3')(conv2)
    flatten = tf.keras.layers.Flatten(name='flatten')(conv3)
    dense1 = tf.keras.layers.Dense(fc1_dims, activation=tf.nn.relu, name='dense1')(flatten)
    # outputs = tf.keras.layers.Dense(num_actions, name='outputs')(dense1)
    return tf.keras.Model(inputs=inputs, outputs=dense1, name=model_name)


def build_cnn_rnn(input_shape, num_actions, agents_names, fc1_dims, fc2_dims, share_weights,
                  embed_dim, rnn_units, dropout, model_name):
    inputs = tf.keras.layers.Input(shape=input_shape)
    conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=(4, 4),
                                   activation=tf.nn.relu, name='conv1')(inputs)
    conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=4, strides=(2, 2),
                                   activation=tf.nn.relu, name='conv2')(conv1)
    conv3 = tf.keras.layers.Conv2D(filters=16, kernel_size=2, strides=(1, 1),
                                   activation=tf.nn.relu, name='conv3')(conv2)
    flatten = tf.keras.layers.Flatten(name='flatten')(conv3)
    dense1 = tf.keras.layers.Dense(fc1_dims, activation=tf.nn.relu, name='dense1')(flatten)
    embed = tf.keras.layers.Embedding(input_dim=fc1_dims, output_dim=embed_dim, name='embed')(dense1)
    lstm = tf.keras.layers.LSTM(units=rnn_units, dropout=dropout, return_sequences=False)(embed)
    dense2 = tf.keras.layers.Dense(fc2_dims, activation=tf.nn.relu, name='dense2')(lstm)

    # outputs = tf.keras.layers.Dense(num_actions, name='output')(dense2)

    return tf.keras.Model(inputs=inputs, outputs=dense2, name=model_name)


def build_mlp(input_shape, num_actions, fc1_dims, fc2_dims, model_name):
    inputs = tf.keras.layers.Input(shape=input_shape)
    flatten = tf.keras.layers.Flatten(name='flatten')(inputs)
    dense1 = tf.keras.layers.Dense(fc1_dims, activation=tf.nn.relu, name='dense1')(flatten)
    dense2 = tf.keras.layers.Dense(fc2_dims, activation=tf.nn.relu, name='dense2')(dense1)
    # outputs = tf.keras.layers.Dense(num_actions, name='outputs')(dense2)
    return tf.keras.Model(inputs=inputs, outputs=dense2, name=model_name)


def build_ma_mlp(input_shape, num_actions, agents_names, fc1_dims, fc2_dims, share_weights, model_name):
    inputs = tf.keras.layers.Input(shape=input_shape)
    if share_weights:
        outputs = {}
        flatten = tf.keras.layers.Flatten(name='flatten')(inputs)
        dense1 = tf.keras.layers.Dense(fc1_dims, activation=tf.nn.relu, name='dense1')(flatten)
        dense2 = tf.keras.layers.Dense(fc2_dims, activation=tf.nn.relu, name='dense2')(dense1)
        for agent_name in agents_names:
            outputs[agent_name] = tf.keras.layers.Dense(num_actions, name=f'output_{agent_name}')(dense2)
    else:
        outputs = {}
        for agent_name in agents_names:
            flatten = tf.keras.layers.Flatten(name=f'flatten_{agent_name}')(inputs)
            dense1 = tf.keras.layers.Dense(fc1_dims, activation=tf.nn.relu, name=f'dense1_{agent_name}')(flatten)
            dense2 = tf.keras.layers.Dense(fc2_dims, activation=tf.nn.relu, name=f'dense2_{agent_name}')(dense1)
            # outputs[agent_name] = tf.keras.layers.Dense(num_actions, name=f'output_{agent_name}')(dense2)

    return tf.keras.Model(inputs=inputs, outputs=dense2, name=model_name)


def build_ma_cnn(input_shape, num_actions, num_agents, fc1_dims, fc2_dims, share_weights, model_name):
    inputs = tf.keras.layers.Input(shape=input_shape)
    if share_weights:
        outputs = {}
        conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=(1, 1),
                                       activation=tf.nn.relu, name='conv1')(inputs)
        maxp1 = tf.keras.layers.MaxPooling2D((3, 3), padding='same', strides=2,
                                             name=f'maxP1')(conv1)
        conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1, 1),
                                       activation=tf.nn.relu, name='conv2')(maxp1)
        maxp2 = tf.keras.layers.MaxPooling2D((3, 3), padding='same', strides=2,
                                             name=f'maxP2')(conv2)
        conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1, 1),
                                       activation=tf.nn.relu, name='conv3')(maxp2)
        maxp3 = tf.keras.layers.MaxPooling2D((3, 3), padding='same', strides=2,
                                             name=f'maxP3')(conv3)
        conv4 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1, 1),
                                       activation=tf.nn.relu, name='conv4')(maxp3)
        maxp4 = tf.keras.layers.MaxPooling2D((3, 3), padding='same', strides=2,
                                             name=f'maxP4')(conv4)
        flatten = tf.keras.layers.Flatten(name='flatten')(maxp4)

        for i in range(num_agents):
            outputs[i] = tf.keras.layers.Dense(fc1_dims, activation=tf.nn.relu, name=f'dense1_agent_{i}')(flatten)

        # for agent_name in agents_names:
        # outputs[agent_name] = tf.keras.layers.Dense(num_actions, name=f'output_{agent_name}')(dense1)

    else:
        outputs = {}
        for i in range(num_agents):
            conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=(3, 3),
                                           activation=tf.nn.relu, name=f'conv1_agent_{i}')(inputs)
            conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=(2, 2),
                                           activation=tf.nn.relu, name=f'conv2_agent_{i}')(conv1)
            conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1),
                                           activation=tf.nn.relu, name=f'conv3_agent_{i}')(conv2)
            flatten = tf.keras.layers.Flatten(name=f'flatten_agent_{i}')(conv3)
            outputs[i] = tf.keras.layers.Dense(fc1_dims, activation=tf.nn.relu, name=f'dense1_agent_{i}')(flatten)
            # outputs[agent_name] = tf.keras.layers.Dense(num_actions, name=f'output_{agent_name}')(dense1)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)

def build_ma_cnn_rnn(input_shape, num_actions, agents_names, fc1_dims, fc2_dims, share_weights,
                     embed_dim, rnn_units, dropout, model_name):
    inputs = tf.keras.layers.Input(shape=input_shape)
    if share_weights:
        outputs = {}
        conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=(4, 4),
                                       activation=tf.nn.relu, name='conv1')(inputs)
        conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=4, strides=(2, 2),
                                       activation=tf.nn.relu, name='conv2')(conv1)
        conv3 = tf.keras.layers.Conv2D(filters=16, kernel_size=2, strides=(1, 1),
                                       activation=tf.nn.relu, name='conv3')(conv2)
        flatten = tf.keras.layers.Flatten(name='flatten')(conv3)
        dense1 = tf.keras.layers.Dense(fc1_dims, activation=tf.nn.relu, name='dense1')(flatten)
        embed = tf.keras.layers.Embedding(input_dim=fc1_dims, output_dim=embed_dim, name='embed')(dense1)
        lstm = tf.keras.layers.LSTM(units=rnn_units, dropout=0.3, return_sequences=False)(embed)
        outputs = tf.keras.layers.Dense(fc2_dims, activation=tf.nn.relu, name='dense2')(lstm)

        # for agent_name in agents_names:
        #     outputs[agent_name] = tf.keras.layers.Dense(num_actions, name=f'output_{agent_name}')(dense2)

    else:
        outputs = {}
        for agent_name in agents_names:
            conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=(3, 3),
                                           activation=tf.nn.relu, name=f'conv1_{agent_name}')(inputs)
            conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=(2, 2),
                                           activation=tf.nn.relu, name=f'conv2_{agent_name}')(conv1)
            conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1),
                                           activation=tf.nn.relu, name=f'conv3_{agent_name}')(conv2)
            flatten = tf.keras.layers.Flatten(name=f'flatten_{agent_name}')(conv3)
            dense1 = tf.keras.layers.Dense(fc1_dims, activation=tf.nn.relu, name=f'dense1_{agent_name}')(flatten)
            embed = tf.keras.layers.Embedding(input_dim=fc1_dims, output_dim=embed_dim, name=f'embed_{agent_name}')(
                dense1)
            lstm = tf.keras.layers.LSTM(units=rnn_units, dropout=dropout, return_sequences=False)(embed)
            outputs[agent_name] = tf.keras.layers.Dense(fc2_dims, activation=tf.nn.relu, name=f'dense2_{agent_name}')(
                lstm)
            # outputs[agent_name] = tf.keras.layers.Dense(num_actions, name=f'output_{agent_name}')(dense2)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)


import numpy as np
import tensorflow as tf

mapping = {}


def register(name):
    def _thunk(func):
        mapping[name] = func
        return func

    return _thunk


def nature_cnn(input_shape, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    print('input shape is {}'.format(input_shape))
    x_input = tf.keras.Input(shape=input_shape, dtype=tf.uint8)
    h = x_input
    h = tf.cast(h, tf.float32) / 255.
    h = conv('c1', nf=32, rf=8, stride=4, activation='relu', init_scale=np.sqrt(2))(h)
    h2 = conv('c2', nf=16, rf=4, stride=2, activation='relu', init_scale=np.sqrt(2))(h)
    h3 = conv('c3', nf=16, rf=3, stride=1, activation='relu', init_scale=np.sqrt(2))(h2)
    h3 = tf.keras.layers.Flatten()(h3)
    h3 = tf.keras.layers.Dense(units=512, kernel_initializer=ortho_init(np.sqrt(2)),
                               name='fc1', activation='relu')(h3)
    network = tf.keras.Model(inputs=[x_input], outputs=[h3])
    return network


@register("mlp")
def mlp(num_layers=2, num_hidden=64, activation=tf.tanh):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """

    def network_fn(input_shape):
        print('input shape is {}'.format(input_shape))
        x_input = tf.keras.Input(shape=input_shape)
        # h = tf.keras.layers.Flatten(x_input)
        h = x_input
        for i in range(num_layers):
            h = tf.keras.layers.Dense(units=num_hidden, kernel_initializer=ortho_init(np.sqrt(2)),
                                      name='mlp_fc{}'.format(i), activation=activation)(h)

        network = tf.keras.Model(inputs=[x_input], outputs=[h])
        return network

    return network_fn


@register("cnn")
def cnn(**conv_kwargs):
    def network_fn(input_shape):
        return nature_cnn(input_shape, **conv_kwargs)

    return network_fn


@register("conv_only")
def conv_only(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], **conv_kwargs):
    '''
    convolutions-only net
    Parameters:
    ----------
    conv:       list of triples (filter_number, filter_size, stride) specifying parameters for each layer.
    Returns:
    function that takes tensorflow tensor as input and returns the output of the last convolutional layer
    '''

    def network_fn(input_shape):
        print('input shape is {}'.format(input_shape))
        x_input = tf.keras.Input(shape=input_shape, dtype=tf.uint8)
        h = x_input
        h = tf.cast(h, tf.float32) / 255.
        with tf.name_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                h = tf.keras.layers.Conv2D(
                    filters=num_outputs, kernel_size=kernel_size, strides=stride,
                    activation='relu', **conv_kwargs)(h)

        network = tf.keras.Model(inputs=[x_input], outputs=[h])
        return network

    return network_fn


def get_network_builder(name):
    """
    If you want to register your own network outside models.py, you just need:

    Usage Example:
    -------------
    from baselines.common.models import register
    @register("your_network_name")
    def your_network_define(**net_kwargs):
        ...
        return network_fn

    """
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Unknown network type: {}'.format(name))


# ('gfootball_impala_cnn')
def gfootball_impala_cnn_rnn(input_shape, num_actions, num_agents, different_output_for_agents, fc1_dims,
                         fc2_dims, share_weights, model_name):
    inputs = tf.keras.layers.Input(shape=input_shape)
    agent_name_inputs = tf.keras.layers.Input(shape=1)
    conv_layers = [(16, 2), (32, 2), (32, 2), (32, 2)]
    conv_out = inputs
    for i, (num_ch, num_blocks) in enumerate(conv_layers):
        conv_out = tf.keras.layers.Conv3D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                          name=f'conv2_{i}')(conv_out)
        conv_out = tf.keras.layers.MaxPooling2D((3, 3), padding='same', strides=2, name=f'maxP_{i}')(conv_out)

        for j in range(num_blocks):
            with tf.name_scope('residual_%d_%d' % (i, j)):
                block_input = conv_out
                conv_out = tf.keras.layers.Activation(tf.nn.relu, name=f'act_{i}_{j}_1')(conv_out)
                conv_out = tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                                  activation=tf.nn.relu, name=f'conv2_{i}_{j}_1')(conv_out)
                conv_out = tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                                  name=f'conv2_{i}_{j}_2')(conv_out)
                conv_out += block_input

    conv_out = tf.keras.layers.Activation(tf.nn.relu)(conv_out)
    conv_out = tf.keras.layers.BatchNormalization()(conv_out)
    conv_out = tf.keras.layers.Flatten()(conv_out)
    conv_out = tf.keras.layers.Dense(fc1_dims, activation=tf.nn.relu, name='dense_lstm')(conv_out)
    conv_out = tf.keras.layers.Embedding(input_dim=fc1_dims, output_dim=96, name='embed')(conv_out)
    conv_out = tf.keras.layers.LSTM(units=256, dropout=0.3, return_sequences=False)(conv_out)


    outputs = tf.keras.layers.Dense(fc1_dims, activation=tf.nn.relu, name=f'dense1')(conv_out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)


# ('gfootball_impala_cnn')
def gfootball_impala_cnn_1(input_shape, num_actions, num_agents, fc1_dims,
                         fc2_dims, share_weights, model_name):
    inputs = tf.keras.layers.Input(shape=input_shape)
    # conv_layers = [(16, 2), (32, 2), (32, 2)]
    conv_layers = [(16, 2), (32, 2), (32, 2), (32, 2)]
    conv_out = inputs
    for i, (num_ch, num_blocks) in enumerate(conv_layers):
        conv_out = tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                          name=f'conv2_{i}')(conv_out)
        conv_out = tf.keras.layers.MaxPooling2D((3, 3), padding='same', strides=2, name=f'maxP_{i}')(conv_out)

        for j in range(num_blocks):
            with tf.name_scope('residual_%d_%d' % (i, j)):
                block_input = conv_out
                conv_out = tf.keras.layers.Activation(tf.nn.relu, name=f'act_{i}_{j}_1')(conv_out)
                conv_out = tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                                  activation=tf.nn.relu, name=f'conv2_{i}_{j}_1')(conv_out)
                conv_out = tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                                  name=f'conv2_{i}_{j}_2')(conv_out)
                conv_out += block_input

    conv_out = tf.keras.layers.Activation(tf.nn.relu)(conv_out)
    conv_out = tf.keras.layers.BatchNormalization()(conv_out)
    conv_out = tf.keras.layers.Flatten()(conv_out)

    outputs = tf.keras.layers.Dense(fc1_dims, activation=tf.nn.relu, name=f'dense1')(conv_out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)



# ('gfootball_impala_cnn')
def gfootball_impala_cnn(input_shape, num_actions, num_agents, fc1_dims,
                         fc2_dims, share_weights, model_name):
    inputs = tf.keras.layers.Input(shape=input_shape)
    agent_name_oh_inputs = tf.keras.layers.Input(shape=(1, num_agents))
    # conv_layers = [(16, 2), (32, 2), (32, 2)]
    conv_layers = [(16, 2), (32, 2), (32, 2), (32, 2)]
    conv_out = inputs
    outputs = {}
    for i, (num_ch, num_blocks) in enumerate(conv_layers):
        conv_out = tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                          name=f'conv2_{i}')(conv_out)
        conv_out = tf.keras.layers.MaxPooling2D((3, 3), padding='same', strides=2, name=f'maxP_{i}')(conv_out)

        for j in range(num_blocks):
            with tf.name_scope('residual_%d_%d' % (i, j)):
                block_input = conv_out
                conv_out = tf.keras.layers.Activation(tf.nn.relu, name=f'act_{i}_{j}_1')(conv_out)
                conv_out = tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                                  activation=tf.nn.relu, name=f'conv2_{i}_{j}_1')(conv_out)
                conv_out = tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                                  name=f'conv2_{i}_{j}_2')(conv_out)
                conv_out += block_input

    conv_out = tf.keras.layers.Activation(tf.nn.relu)(conv_out)
    conv_out = tf.keras.layers.BatchNormalization()(conv_out)
    conv_out = tf.keras.layers.Flatten()(conv_out)
    agent_name_oh_flatted = tf.keras.layers.Flatten()(agent_name_oh_inputs)

    conv_out = tf.keras.layers.concatenate([agent_name_oh_flatted, conv_out])

    outputs = tf.keras.layers.Dense(fc1_dims, activation=tf.nn.relu, name=f'dense1')(conv_out)

    return tf.keras.Model(inputs=[inputs, agent_name_oh_inputs], outputs=outputs, name=model_name)


def impala_fp(input_shape, fc1_dims, model_name, num_agents, num_actions, num_extra_data=2):
    inputs = tf.keras.layers.Input(shape=input_shape)
    agent_name_oh_inputs = tf.keras.layers.Input(shape=(1, num_agents))
    FingerPrint_policy_inputs = tf.keras.layers.Input(shape=(num_agents - 1, num_actions))  # other agents policy
    FingerPrint_eps_time_inputs = tf.keras.layers.Input(shape=(num_agents - 1, num_extra_data))

    agent_name_oh_flatted = tf.keras.layers.Flatten()(agent_name_oh_inputs)
    FingerPrint_policy_Flat = tf.keras.layers.Flatten()(FingerPrint_policy_inputs)
    FingerPrint_eps_time_Flat = tf.keras.layers.Flatten()(FingerPrint_eps_time_inputs)

    # conv_layers = [(16, 2), (32, 2), (32, 2)]
    conv_layers = [(16, 2), (32, 2), (32, 2), (32, 2)]
    conv_out = inputs

    for i, (num_ch, num_blocks) in enumerate(conv_layers):
        conv_out = tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                          name=f'conv_{i}')(conv_out)
        conv_out = tf.keras.layers.MaxPooling2D((3, 3), padding='same', strides=2, name=f'maxP_{i}')(conv_out)

        for j in range(num_blocks):
            with tf.name_scope('residual_%d_%d' % (i, j)):
                block_input = conv_out
                conv_out = tf.keras.layers.Activation(tf.nn.relu, name=f'act_{i}_{j}_1')(conv_out)
                conv_out = tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                                  activation=tf.nn.relu, name=f'conv_{i}_{j}_1')(conv_out)
                conv_out = tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                                  name=f'conv_{i}_{j}_2')(conv_out)
                conv_out += block_input

    conv_out = tf.keras.layers.Activation(tf.nn.relu)(conv_out)
    conv_out = tf.keras.layers.BatchNormalization()(conv_out)
    conv_out = tf.keras.layers.Flatten()(conv_out)
    conv_out = tf.keras.layers.Dense(fc1_dims, activation=tf.nn.relu, name=f'dense1')(conv_out)
    # conv_out = tf.keras.layers.BatchNormalization()(conv_out)

    outputs = tf.keras.layers.concatenate([conv_out, FingerPrint_policy_Flat, FingerPrint_eps_time_Flat, agent_name_oh_flatted])
    outputs = tf.keras.layers.Flatten()(outputs)
    outputs = tf.keras.layers.Dense(fc1_dims, activation=tf.nn.relu, name=f'dense1_after_fps')(outputs)
    # outputs = tf.keras.layers.BatchNormalization()(outputs)

    return tf.keras.Model(inputs=[inputs, agent_name_oh_inputs, FingerPrint_policy_inputs, FingerPrint_eps_time_inputs],
                          outputs=outputs, name=model_name)



def impala_fp_rnn(input_shape, fc1_dims, fc2_dims, rnn_unit, num_actions, model_name, num_agents, n_step, num_extra_data=2):
    inputs = tf.keras.layers.Input(shape=input_shape)
    done_Input = tf.keras.layers.Input(shape=(n_step, 1))

    agent_name_oh_inputs = tf.keras.layers.Input(shape=(1, num_agents))
    agent_name_oh_flatted = tf.keras.layers.Flatten()(agent_name_oh_inputs)

    FingerPrint_policy_inputs = tf.keras.layers.Input(shape=(num_agents - 1, num_actions))  # other agents policy
    FingerPrint_eps_time_inputs = tf.keras.layers.Input(shape=(num_agents - 1, num_extra_data))
    concat_fps = tf.keras.layers.concatenate([FingerPrint_policy_inputs, FingerPrint_eps_time_inputs], name='concat1')
    fps_flat = tf.keras.layers.Flatten()(concat_fps)
    fps_dense = tf.keras.layers.Dense(fc1_dims//2, activation=tf.nn.relu, name=f'fps_dense')(fps_flat)

    # conv_layers = [(16, 2), (32, 2)]
    # conv_layers = [(16, 2), (32, 2), (32, 2)]
    conv_layers = [(16, 2), (32, 2), (32, 2), (32, 2)]
    conv_out = inputs

    for i, (num_ch, num_blocks) in enumerate(conv_layers):
        conv_out = tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                          name=f'conv2_{i}')(conv_out)
        conv_out = tf.keras.layers.MaxPooling2D((3, 3), padding='same', strides=2, name=f'maxP_{i}')(conv_out)

        for j in range(num_blocks):
            with tf.name_scope('residual_%d_%d' % (i, j)):
                block_input = conv_out
                conv_out = tf.keras.layers.Activation(tf.nn.relu, name=f'act_{i}_{j}_1')(conv_out)
                conv_out = tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                                  activation=tf.nn.relu, name=f'conv2_{i}_{j}_1')(conv_out)
                conv_out = tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                                  name=f'conv2_{i}_{j}_2')(conv_out)
                conv_out += block_input

    conv_out = tf.keras.layers.Activation(tf.nn.relu)(conv_out)
    conv_out = tf.keras.layers.BatchNormalization()(conv_out)
    conv_out = tf.keras.layers.Flatten()(conv_out)
    conv_out = tf.keras.layers.Dense(fc1_dims, activation=tf.nn.relu, name=f'dense1_agent_a')(conv_out)
    conv_out = tf.keras.layers.BatchNormalization()(conv_out)

    outputs = tf.keras.layers.concatenate([conv_out, agent_name_oh_flatted, fps_dense], name='concat2')
    outputs = tf.keras.layers.Flatten()(outputs)
    outputs = tf.keras.layers.Dense(fc2_dims, activation=tf.nn.relu, name=f'dense1_after_fps')(outputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)

    reshaped = tf.reshape(outputs, shape=(-1, n_step, fc2_dims))

    masked = reshaped * (1. - done_Input)
    LSTM_masked = tf.keras.layers.Masking(mask_value=0., input_shape=(n_step, fc2_dims))(masked)
    outputs = tf.keras.layers.LSTM(units=rnn_unit, dropout=0.3, return_sequences=False)(LSTM_masked)

    return tf.keras.Model(inputs=[inputs, agent_name_oh_inputs, FingerPrint_policy_inputs, FingerPrint_eps_time_inputs,
                                  done_Input], outputs=outputs, name=model_name)



def impala_fp_rnn2(input_shape, fc1_dims, fc2_dims, model_name, num_agents, n_step, num_extra_data=2):
    inputs = tf.keras.layers.Input(shape=input_shape)
    agent_name_oh_inputs = tf.keras.layers.Input(shape=(1, num_agents))
    FingerPrint_policy_inputs = tf.keras.layers.Input(shape=(num_agents - 1, fc1_dims))  # other agents policy
    FingerPrint_eps_time_inputs = tf.keras.layers.Input(shape=(num_agents - 1, num_extra_data))
    done_Input = tf.keras.layers.Input(shape=(n_step, 1))

    conv_layers = [(16, 2), (32, 2)]
    # conv_layers = [(16, 2), (32, 2), (32, 2)]
    # conv_layers = [(16, 2), (32, 2), (32, 2), (32, 2)]
    conv_out = inputs
    for i, (num_ch, num_blocks) in enumerate(conv_layers):
        conv_out = tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                          name=f'conv2_{i}')(conv_out)
        conv_out = tf.keras.layers.MaxPooling2D((3, 3), padding='same', strides=2, name=f'maxP_{i}')(conv_out)

        for j in range(num_blocks):
            with tf.name_scope('residual_%d_%d' % (i, j)):
                block_input = conv_out
                conv_out = tf.keras.layers.Activation(tf.nn.relu, name=f'act_{i}_{j}_1')(conv_out)
                conv_out = tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                                  activation=tf.nn.relu, name=f'conv2_{i}_{j}_1')(conv_out)
                conv_out = tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                                  name=f'conv2_{i}_{j}_2')(conv_out)
                conv_out += block_input

    conv_out = tf.keras.layers.Activation(tf.nn.relu)(conv_out)
    conv_out = tf.keras.layers.BatchNormalization()(conv_out)
    conv_out = tf.keras.layers.Flatten()(conv_out)
    conv_out = tf.keras.layers.Dense(fc1_dims, activation=tf.nn.relu, name=f'dense1_agent_a')(conv_out)
    conv_out = tf.keras.layers.BatchNormalization()(conv_out)

    agent_name_oh_flatted = tf.keras.layers.Flatten()(agent_name_oh_inputs)
    FingerPrint_policy_Flat = tf.keras.layers.Flatten()(FingerPrint_policy_inputs)
    FingerPrint_eps_time_Flat = tf.keras.layers.Flatten()(FingerPrint_eps_time_inputs)

    concated = tf.keras.layers.concatenate([conv_out, FingerPrint_policy_Flat, FingerPrint_eps_time_Flat, agent_name_oh_flatted])
    flat = tf.keras.layers.Flatten()(concated)
    outputs = tf.keras.layers.Dense(fc2_dims, activation=tf.nn.relu, name=f'dense1_after_fps')(flat)
    reshaped = tf.reshape(outputs, shape=(-1, n_step, fc2_dims))

    masked = reshaped * (1. - done_Input)
    LSTM_masked = tf.keras.layers.Masking(mask_value=0., input_shape=(n_step, fc2_dims))(masked)
    outputs = tf.keras.layers.LSTM(units=fc1_dims, dropout=0.3, return_sequences=False)(LSTM_masked)

    return tf.keras.Model(inputs=[inputs, agent_name_oh_inputs, FingerPrint_policy_inputs, FingerPrint_eps_time_inputs,
                                  done_Input], outputs=outputs, name=model_name)


def FP_model(fc_size, num_agents, fp_size=512, extra_data=2):
    agent_policy_input = tf.keras.layers.Input(shape=fp_size) # agent a policy
    FingerPrint_policy_inputs = tf.keras.layers.Input(shape=(num_agents-1, fp_size))  # other agents policy
    FingerPrint_eps_time_inputs = tf.keras.layers.Input(shape=(num_agents-1, extra_data))

    FingerPrint_policy_Flat = tf.keras.layers.Flatten()(FingerPrint_policy_inputs)
    FingerPrint_eps_time_Flat = tf.keras.layers.Flatten()(FingerPrint_eps_time_inputs)

    concated = tf.keras.layers.concatenate([agent_policy_input, FingerPrint_policy_Flat, FingerPrint_eps_time_Flat])
    flat = tf.keras.layers.Flatten()(concated)
    outputs = tf.keras.layers.Dense(fc_size, activation=tf.nn.relu, name=f'dense1')(flat)

    model_FP = tf.keras.Model(inputs=[agent_policy_input, FingerPrint_policy_inputs, FingerPrint_eps_time_inputs],
                              outputs=outputs, name='FP_model')

    return model_FP

#
# num_agents = 5
# batch_size = 8
# n_step = 16
# fc1_dims = 256
# fc2_dims = 512
# rnn_unit = 256
# agents_names = [0, 1, 2, 3, 4] #, 'agent_3', 'agent_4']
#
# # tf.device('cpu')
# import numpy as np
# obs_hist = np.random.normal(0.5, 2.0, (batch_size*n_step, 51, 40, 16))
# obs_hist = np.array(obs_hist, dtype=np.float32)
# done_hist = np.zeros((batch_size, n_step))
# done_hist[:, -1] = 1
#
# Y = np.concatenate((np.zeros(shape=(25), dtype=np.int32), np.ones(shape=(25), dtype=np.int32),
#                     np.ones(shape=(25), dtype=np.int32)*2, np.ones(shape=(25), dtype=np.int32)*3))
# # print(Y.shape)
#
# obs = np.random.normal(0.5, 2.0, (batch_size, 51, 40, 16))
# input_shape = obs[0].shape
# print(f'input_shape is {input_shape}')
# # shared_network = MAMLP(num_actions=19, num_agents=5, share_weights=True)
# # shared_network = gfootball_impala_cnn(input_shape, 4, 4, fc_size, fc_size, True, 'IMPALA_cnn')
# # shared_network = build_ma_cnn(input.shape[1:], 4, agents_names, 512, 512, True, 'cnn_rnn')
# # shared_network = impala_fp_rnn(input_shape, fc_size, 'IMPALA_FP', num_agents, extra_data=2)
# shared_network = impala_fp_rnn(input_shape, fc1_dims, fc2_dims, rnn_unit, 'IMPALA_FP_RNN', num_agents, n_step, num_extra_data=2)
#
# opt = tf.optimizers.Adam(lr=0.01)
# # print(shared_network.trainable_variables)
# shared_network.compile(optimizer=opt)
# shared_network.summary()
# tf.keras.utils.plot_model(shared_network, to_file='./IMPALA_FP_RNN.png')
# print(f'shared network output shape is {shared_network.output_shape[-1]}')
# print(f'shared network input shape is {shared_network.input_shape}')
#
# print(f'obs_hist shape is {obs_hist.shape}')
# one_hot_agent = np.expand_dims(tf.one_hot(agents_names, len(agents_names), dtype=tf.float32)[0].numpy(), axis=0)
# tile_time = batch_size*n_step
# agent_one_hot = np.tile(np.expand_dims(one_hot_agent, axis=0), (tile_time,1,1))
# agent_one_hot = np.array(agent_one_hot, dtype=np.float32)
# print(f'agent_one_hot shape is {agent_one_hot.shape}')
#
# dummy_fps = np.tile(np.zeros((1, num_agents - 1, fc1_dims), dtype=np.float32), (batch_size*n_step,1,1))
# print(f'dummy_fps shape is {dummy_fps.shape}')
#
# dummy_extra_data = np.tile(np.zeros((1, num_agents - 1, 2), dtype=np.float32), (batch_size*n_step,1,1))
# print(f'dummy_extra_data shape is {dummy_extra_data.shape}')
#
# done_mask = np.zeros((batch_size, n_step, 1), dtype=np.float32)
# done_mask[:,-1] = done_mask[:,-1] + 1.
# print(f'done_mask shape is {done_mask.shape}')
# # print(done_mask)
#
# out = shared_network({0: obs_hist, 1: agent_one_hot, 2: dummy_fps, 3: dummy_extra_data , 4:done_mask})
# print(out)
#
# obs_hist_shape = obs_hist.shape
# print(obs_hist_shape)
#
# obs_hist = np.reshape(obs_hist, (obs_hist_shape[0]*obs_hist_shape[1], *obs_hist_shape[2:]))
# print(obs_hist.shape)
#
# done_hist = np.reshape(done_hist, (obs_hist_shape[0]*obs_hist_shape[1]))
# print(done_hist)
# agents_name = [[0.0] for _ in range(obs_hist.shape[0])]
# agents_name = np.asarray(agents_name)
# print(agents_name.shape)
#
# IMPALA_out = shared_network({0: obs_hist, 1: agents_name})
#
# IMPALA_out = IMPALA_out.numpy()
# print(IMPALA_out.shape)
#
#
#
# model_FP = FP_model(fc_size=512, num_agents=2, fp_size=512, extra_data=2)
# model_FP.summary()
# tf.keras.utils.plot_model(model_FP, to_file='./model_FP.png')






# for i in range(n_step):
#     IMPALA_out[i, :] = 0. if done_hist[i] == 1 else IMPALA_out[i, :]
#
# print(IMPALA_out[3])
# #
# IMPALA_out = tf.reshape(IMPALA_out, (obs_hist_shape[0], obs_hist_shape[1], -1))
# print(IMPALA_out.shape)
# print(IMPALA_out)
#
#
# LSTM_Input = tf.keras.layers.Input(IMPALA_out[0].shape)
# print(LSTM_Input)
# LSTM_masked = tf.keras.layers.Masking(mask_value=0., input_shape=(n_step, fc_size))(LSTM_Input)
# lstm = tf.keras.layers.LSTM(units=32, dropout=0.3, return_sequences=False)(LSTM_masked)
# print(lstm)
#
# model = tf.keras.Model(inputs=LSTM_Input, outputs=lstm, name='PO_LSTM')
# model.summary()
# tf.keras.utils.plot_model(model, to_file='./PO_LSTM_model.png')
# model_out = model(IMPALA_out)
# print(model_out)
#
#
# vf = tf.keras.layers.Dense(1, name='vf')(lstm)
# model_vf = tf.keras.Model(inputs=LSTM_Input, outputs=vf, name='vf')
# model_vf.summary()
# vf_out = model_vf(IMPALA_out)
#
# print(vf_out)
#
#
# param = [v for v in shared_network.trainable_variables if v.name.__contains__('agent_0')]
# # print(param)
# y = {}


@tf.function
def train(batch_x, batch_y, shared_network, opt):
    with tf.GradientTape() as tape:
        q_eval_arr = shared_network(batch_x)['agent_0']
        # print('q_eval_arr ', q_eval_arr)
        one_hot = tf.one_hot(batch_y, 4)
        # print(one_hot)
        q_t_selected = tf.reduce_sum(q_eval_arr * one_hot, 1)
        target_q_values = shared_network(batch_x + np.random.normal(0, 1, batch_x.shape))['agent_0']
        max_target_q_values = tf.reduce_max(target_q_values, axis=1)
        td_error = q_t_selected - max_target_q_values
        # print('td_error ', td_error)
        errors = huber_loss(td_error)
        loss = tf.reduce_mean(errors)

    # param = shared_network.trainable_variables
    param = [v for v in shared_network.trainable_variables if v.name.__contains__('agent_0')]
    gradients_of_network = tape.gradient(loss, param)
    opt.apply_gradients(zip(gradients_of_network, param))

# batch_x = input[0:4]
# print('batch_x ', batch_x.shape)
# batch_y = Y[0:4]
# print('batch_y ', batch_y.shape)
# train(batch_x, batch_y)
# print(shared_network(batch_x))

# for i in range(input.shape[0]-5):
#     batch_x = input[i:i+4]
#     batch_y = Y[i:i+4]
#     train(batch_x, batch_y, shared_network, opt)
#     print(shared_network(batch_x))


# shared_network.summary()
# tf.keras.utils.plot_model(shared_network, to_file='./model.png')
