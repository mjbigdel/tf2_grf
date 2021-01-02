from distutils.command.config import config

import tensorflow as tf
import numpy as np

from common.schedules import LinearSchedule

from basic_dqn_FP_RNN_1.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

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


def init_network(config):
    """

    :param config: config object containing all parameters and hyper-parameters
        resnet style with conv_layers = [(16, 2), (32, 2), (32, 2), (32, 2)]
    :return: Tensorflow Model
    """
    kernel_init = tf.keras.initializers.Orthogonal(gain=1.0)
    bias_init = tf.keras.initializers.Constant(0.0)
    TdL = tf.keras.layers.TimeDistributed

    inputs_obs = tf.keras.Input(shape=(None, *config.obs_shape), name='0')  # , name='1'
    inputs_agent_name_oh = tf.keras.Input(shape=(config.num_agents,), name='1')
    fp_shape = (config.num_agents-1)*config.num_actions+config.num_extra_data
    inputs_fps = tf.keras.Input(shape=(None, fp_shape), name='2')

    done_Input = tf.keras.Input(shape=(None,), name='3')
    masking = tf.cast(1. - done_Input, dtype=tf.bool)

    conv_layers = [(16, 2), (32, 2), (32, 2), (32, 2)]
    conv_out = inputs_obs
    # conv_out = tf.cast(conv_out, tf.float32) / 255.
    for i, (num_ch, num_blocks) in enumerate(conv_layers):
        conv_out = TdL(tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                          name=f'conv2_{i}', kernel_initializer=kernel_init,
                                          bias_initializer=bias_init))(conv_out)
        conv_out = TdL(tf.keras.layers.MaxPooling2D((3, 3), padding='same', strides=2, name=f'maxP_{i}'))(conv_out)
        conv_out = TdL(tf.keras.layers.BatchNormalization())(conv_out)
        
        for j in range(num_blocks):
            with tf.name_scope('residual_%d_%d' % (i, j)):
                block_input = conv_out      
                conv_out = TdL(tf.keras.layers.Activation(tf.nn.relu, name=f'act_{i}_{j}_1'))(conv_out)                          
                conv_out = TdL(tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                                  name=f'conv2_{i}_{j}_1',
                                                  kernel_initializer=kernel_init, bias_initializer=bias_init))(conv_out)
                conv_out = TdL(tf.keras.layers.BatchNormalization())(conv_out)
                conv_out = TdL(tf.keras.layers.Activation(tf.nn.relu, name=f'act_{i}_{j}_2'))(conv_out)
                conv_out = TdL(tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                                  name=f'conv2_{i}_{j}_2', kernel_initializer=kernel_init,
                                                  bias_initializer=bias_init))(conv_out)
                conv_out = TdL(tf.keras.layers.BatchNormalization())(conv_out)
                conv_out = TdL(tf.keras.layers.Activation(tf.nn.relu, name=f'act_{i}_{j}_3'))(conv_out)
                conv_out += block_input

    conv_out = TdL(tf.keras.layers.Activation(tf.nn.relu))(conv_out)
    conv_out = TdL(tf.keras.layers.BatchNormalization())(conv_out)
    conv_out = TdL(tf.keras.layers.Flatten())(conv_out)
    # conv_out = TdL(tf.keras.layers.concatenate([conv_out, inputs_agent_name_oh]))
    conv_out = TdL(tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu,
                                 kernel_initializer=kernel_init, bias_initializer=bias_init,
                                 name='dense1'))(conv_out)

    fps_dense = TdL(tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu,
                              kernel_initializer=kernel_init, bias_initializer=bias_init,
                              name='dense_fps'))(inputs_fps)

    outputs = tf.keras.layers.concatenate([conv_out, fps_dense])

    # outputs = tf.reshape(outputs, shape=(-1, config.n_steps, outputs.shape[-1]))
    outputs = tf.keras.layers.LSTM(units=512, dropout=0.25, return_sequences=True)(inputs=outputs, mask=masking)
    outputs = tf.keras.layers.BatchNormalization()(outputs)

    return tf.keras.Model(inputs=[inputs_obs, inputs_agent_name_oh, inputs_fps, done_Input], outputs=outputs, name=config.network)



def init_network_2(config):
    """

    :param config: config object containing all parameters and hyper-parameters
        resnet style with conv_layers = [(16, 2), (32, 2), (32, 2), (32, 2)]
    :return: Tensorflow Model
    """
    kernel_init = tf.keras.initializers.Orthogonal(gain=1.0)
    bias_init = tf.keras.initializers.Constant(0.0)
    TdL = tf.keras.layers.TimeDistributed

    inputs_obs = tf.keras.Input(shape=(None, *config.obs_shape), name='0')  # , name='1'
    inputs_agent_name_oh = tf.keras.Input(shape=(config.num_agents,), name='1')
    fp_shape = (config.num_agents-1)*config.num_actions+config.num_extra_data
    inputs_fps = tf.keras.Input(shape=(None, fp_shape), name='2')

    done_Input = tf.keras.Input(shape=(None,), name='3')
    masking = tf.cast(1. - done_Input, dtype=tf.bool)

    # conv_layers = [(16, 2), (32, 2), (32, 2), (32, 2)]
    conv_layers = [(16, 2), (32, 2), (32, 2)]
    conv_out = inputs_obs
    # conv_out = tf.cast(conv_out, tf.float32) / 255.
    for i, (num_ch, num_blocks) in enumerate(conv_layers):
        conv_out = tf.keras.layers.ConvLSTM2D(filters=num_ch, kernel_size=3, strides=1, padding='same', return_sequences=True,
                                              name=f'conv2_{i}', kernel_initializer=kernel_init,
                                              bias_initializer=bias_init)(conv_out, mask=masking)
        print(f'conv2_0.shape {conv_out.shape}')
        conv_out = tf.keras.layers.MaxPooling3D((1, 3, 3), padding='same', strides=(1, 2, 2), name=f'maxP_{i}')(conv_out)

        for j in range(num_blocks):
            with tf.name_scope('residual_%d_%d' % (i, j)):
                block_input = conv_out
                conv_out = tf.keras.layers.Activation(tf.nn.relu, name=f'act_{i}_{j}_1')(conv_out)
                conv_out = tf.keras.layers.ConvLSTM2D(filters=num_ch, kernel_size=3, strides=1, padding='same', return_sequences=True,
                                                      activation=tf.nn.relu, name=f'conv2_{i}_{j}_1',
                                                      kernel_initializer=kernel_init,
                                                      bias_initializer=bias_init)(conv_out, mask=masking)
                conv_out = tf.keras.layers.ConvLSTM2D(filters=num_ch, kernel_size=3, strides=1, padding='same', return_sequences=True,
                                                      name=f'conv2_{i}_{j}_2', kernel_initializer=kernel_init,
                                                      bias_initializer=bias_init)(conv_out, mask=masking)
                conv_out += block_input

    conv_out = tf.keras.layers.Activation(tf.nn.relu)(conv_out)
    conv_out = tf.keras.layers.BatchNormalization()(conv_out)
    conv_out = TdL((tf.keras.layers.Flatten()))(conv_out)
    # print(f'np.prod(conv_out.shape[2:]) is {np.prod(conv_out.shape[2:])}')
    # conv_out = tf.keras.layers.Reshape((config.n_steps, np.prod(conv_out.shape[2:])))(conv_out)
    # print(f'conv_out.shape is {conv_out.shape}')
    
    fps_dense = TdL(tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu,
                            kernel_initializer=kernel_init, bias_initializer=bias_init,
                            name='dense_fps'))(inputs_fps)

    outputs = tf.keras.layers.concatenate([conv_out, fps_dense])
    outputs = TdL(tf.keras.layers.Dense(512))(outputs)
    # outputs = tf.keras.layers.concatenate([outputs, inputs_agent_name_oh])
    return tf.keras.Model(inputs=[inputs_obs, inputs_agent_name_oh, inputs_fps, done_Input], outputs=outputs, name=config.network)



