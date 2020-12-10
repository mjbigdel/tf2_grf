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


def init_network(config):
    """

    :param config: config object containing all parameters and hyper-parameters
        resnet style with conv_layers = [(16, 2), (32, 2), (32, 2), (32, 2)]
    :return: Tensorflow Model
    """
    kernel_init = tf.keras.initializers.Orthogonal(gain=1.0)
    bias_init = tf.keras.initializers.Constant(0.0)

    # inputs
    inputs_obs = tf.keras.layers.Input(shape=config.obs_shape)
    inputs_agent_name_oh = tf.keras.layers.Input(shape=(1, config.num_agents))
    inputs_fps = tf.keras.layers.Input(shape=(config.num_agents-1, config.num_actions))
    fps_flat = tf.keras.layers.Flatten()(inputs_fps)

    conv_layers = [(16, 2), (32, 2), (32, 2), (32, 2)]
    conv_out = inputs_obs
    # conv_out = tf.cast(conv_out, tf.float32) / 255.
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
    # conv_out = tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu,
    #                                  kernel_initializer=kernel_init, bias_initializer=bias_init,
    #                                  name='dense1')(conv_out)

    agent_name_oh_flatted = tf.keras.layers.Flatten()(inputs_agent_name_oh)

    outputs = tf.keras.layers.concatenate([agent_name_oh_flatted, conv_out, fps_flat])

    outputs = tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu,
                                    kernel_initializer=kernel_init, bias_initializer=bias_init,
                                    name='dense2')(outputs)

    return tf.keras.Model(inputs=[inputs_obs, inputs_agent_name_oh, inputs_fps], outputs=outputs, name=config.network)
