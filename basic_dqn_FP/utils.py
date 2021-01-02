from distutils.command.config import config

import tensorflow as tf
from basic_dqn_FP.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from common.schedules import LinearSchedule


def init_replay_memory(config):
    """

    :param config: config object containing all parameters and hyper-parameters
    :return: replay_buffer, beta_schedule
    """
    if config.prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(config.buffer_size, alpha=config.prioritized_replay_alpha)
        if config.prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = config.num_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=config.prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(config.buffer_size)
        beta_schedule = None
    return replay_buffer, beta_schedule


def init_cnn(config):
    """

    :param config: config object containing all parameters and hyper-parameters
        resnet style with conv_layers = [(16, 2), (32, 2), (32, 2), (32, 2)]
    :return: Tensorflow Keras Model
    """
    kernel_init = tf.keras.initializers.Orthogonal(gain=1.0)
    bias_init = tf.keras.initializers.Constant(0.0)

    # inputs
    inputs_obs = tf.keras.layers.Input(shape=config.obs_shape)
    inputs_fps = tf.keras.layers.Input(shape=(config.fp_shape,))

    # conv_layers = [(16, 2), (32, 2), (32, 2), (32, 2)]
    conv_layers = [(16, 2), (32, 2), (32, 2)]
    conv_out = inputs_obs
    for i, (num_ch, num_blocks) in enumerate(conv_layers):
        conv_out = tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                          name=f'conv2_{i}', kernel_initializer=kernel_init,
                                          bias_initializer=bias_init)(conv_out)
        conv_out = tf.keras.layers.MaxPooling2D((3, 3), padding='same', strides=2, name=f'maxP_{i}')(conv_out)

        for j in range(num_blocks):
            with tf.name_scope('residual_%d_%d' % (i, j)):
                block_input = conv_out
                conv_out = tf.keras.layers.Activation(tf.nn.relu, name=f'act_{i}_{j}_1')(conv_out)
                conv_out = tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                                  activation=tf.nn.relu, name=f'conv2_{i}_{j}_1',
                                                  kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_out)
                conv_out = tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                                  name=f'conv2_{i}_{j}_2', kernel_initializer=kernel_init,
                                                  bias_initializer=bias_init)(conv_out)
                conv_out += block_input

    conv_out = tf.keras.layers.Activation(tf.nn.relu)(conv_out)
    conv_out = tf.keras.layers.BatchNormalization()(conv_out)
    conv_out = tf.keras.layers.Flatten()(conv_out)
    conv_out = tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu,
                                  kernel_initializer=kernel_init, bias_initializer=bias_init,
                                  name='dense1')(conv_out)

    fps_dense = tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu,
                              kernel_initializer=kernel_init, bias_initializer=bias_init,
                              name='dense_fps')(inputs_fps)

    outputs = tf.keras.layers.concatenate([conv_out, fps_dense])

    outputs = tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu,
                                    kernel_initializer=kernel_init, bias_initializer=bias_init,
                                    name='dense3')(outputs)

    return tf.keras.Model(inputs=[inputs_obs, inputs_fps], outputs=outputs, name=config.network)


def init_tdcnn_rnn(config):
    """

    :param config: config object containing all parameters and hyper-parameters
        resnet style with conv_layers = [(16, 2), (32, 2), (32, 2), (32, 2)]
    :return: Tensorflow Keras Model
    """
    kernel_init = tf.keras.initializers.Orthogonal(gain=1.0)
    bias_init = tf.keras.initializers.Constant(0.0)
    TdL = tf.keras.layers.TimeDistributed

    # inputs
    inputs_obs = tf.keras.layers.Input(shape=(None, *config.obs_shape))
    inputs_fps = tf.keras.layers.Input(shape=(None, config.fp_shape))

    # conv_layers = [(16, 2), (32, 2), (32, 2), (32, 2)]
    conv_layers = [(16, 2), (32, 2), (32, 2)]
    conv_out = inputs_obs
    for i, (num_ch, num_blocks) in enumerate(conv_layers):
        conv_out = TdL(conv(num_ch, 3, 1, 'same', f'{i}', kernel_init, bias_init))(conv_out)
        conv_out = TdL(tf.keras.layers.MaxPooling2D((3, 3), padding='same', strides=2, name=f'maxP_{i}'))(conv_out)

        for j in range(num_blocks):
            with tf.name_scope('residual_%d_%d' % (i, j)):
                block_input = conv_out
                conv_out = tf.keras.layers.Activation(tf.nn.relu, name=f'act_{i}_{j}_1')(conv_out)

                conv_out = TdL(conv(num_ch, 3, 1, 'same', f'{i}_{j}_1', kernel_init, bias_init))(conv_out)
                conv_out = TdL(conv(num_ch, 3, 1, 'same', f'{i}_{j}_2', kernel_init, bias_init))(conv_out)

                conv_out += block_input

    conv_out = tf.keras.layers.Activation(tf.nn.relu)(conv_out)
    conv_out = tf.keras.layers.BatchNormalization()(conv_out)
    conv_out = TdL(tf.keras.layers.Flatten())(conv_out)
    conv_out = TdL(dense(512, 'dense1', kernel_init, bias_init))(conv_out)

    fps_dense = TdL(dense(128, 'dense_fps', kernel_init, bias_init))(inputs_fps)

    outputs = tf.keras.layers.concatenate([conv_out, fps_dense])

    # outputs = tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu,
    #                                 kernel_initializer=kernel_init, bias_initializer=bias_init,
    #                                 name='dense3')(outputs)

    outputs = tf.keras.layers.LSTM(units=512, dropout=0.25, return_sequences=False)(inputs=outputs)

    return tf.keras.Model(inputs=[inputs_obs, inputs_fps], outputs=outputs, name=config.network)


def conv(num_ch, ks, st, pad, name, k_init, b_init):
    return tf.keras.layers.Conv2D(filters=num_ch, kernel_size=ks, strides=st, padding=pad, name=f'conv2_{name}',
                                  kernel_initializer=k_init, bias_initializer=b_init)


def dense(units, name, k_init, b_init, act=tf.keras.activations.relu):
    return tf.keras.layers.Dense(units=units, activation=act, kernel_initializer=k_init,
                                 bias_initializer=b_init, name=name)