import tensorflow as tf
import numpy as np


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


def build_q_func(network, hiddens=[256], dueling=True, layer_norm=False):
    def q_func_builder(input_shape, num_actions, agent_ids):
        # the sub Functional model which does not include the top layer.
        model = network

        # wrapping the sub Functional model with layers that compute action scores into another Functional model.
        latent = model.outputs
        if len(latent) > 1:
            if latent[1] is not None:
                raise NotImplementedError("DQN is not compatible with recurrent policies yet")
        latent = latent[0]

        latent = tf.keras.layers.Flatten()(latent)

        with tf.name_scope("action_value"):
            action_out = latent
            for hidden in hiddens:
                action_out = tf.keras.layers.Dense(units=hidden, activation=None, kernel_initializer=ortho_init(1.0),
                                  bias_initializer=tf.keras.initializers.Constant(1.0))(action_out)
                if layer_norm:
                    action_out = tf.keras.layers.LayerNormalization(center=True, scale=True)(action_out)
                action_out = tf.nn.relu(action_out)
            action_scores = []
            for a in agent_ids:
                action_scores.append(tf.keras.layers.Dense(units=num_actions, activation=None,
                                                           kernel_initializer=ortho_init(1.0),
                                                           bias_initializer=tf.keras.initializers.Constant(1.0),
                                                           name=f'action_score_agent_{a}')(action_out))

        if dueling:
            with tf.name_scope("state_value"):
                state_out = latent
                for hidden in hiddens:
                    state_out = tf.keras.layers.Dense(units=hidden, activation=None,
                                                      kernel_initializer=ortho_init(1.0),
                                                      bias_initializer=tf.keras.initializers.Constant(1.0)
                                                      )(state_out)
                    if layer_norm:
                        state_out = tf.keras.layers.LayerNormalization(center=True, scale=True)(state_out)
                    state_out = tf.nn.relu(state_out)
                state_score = []
                for a in agent_ids:
                    state_score.append(tf.keras.layers.Dense(units=1, activation=None,
                                                             kernel_initializer=ortho_init(1.0),
                                                             bias_initializer=tf.keras.initializers.Constant(1.0),
                                                             name=f'state_score_agent_{a}')(state_out))

            q_out = []
            for a in agent_ids:
                action_scores_mean = tf.reduce_mean(action_scores[a], 1)
                action_scores_centered = action_scores[a] - tf.expand_dims(action_scores_mean, 1)
                q_out.append(state_score[a] + action_scores_centered)
        else:
            q_out = action_scores

        return tf.keras.Model(inputs=model.inputs, outputs=q_out)

    return q_func_builder


def impala_fp(input_shape, num_agents, model_name, num_actions, num_extra_data=2):
    inputs = tf.keras.layers.Input(shape=input_shape)
    agent_name_oh_inputs = tf.keras.layers.Input(shape=(1, num_agents))
    FingerPrint_policy_inputs = tf.keras.layers.Input(shape=(num_agents - 1, num_actions))  # other agents policy
    FingerPrint_eps_time_inputs = tf.keras.layers.Input(shape=(num_extra_data))

    agent_name_oh_flatted = tf.keras.layers.Flatten()(agent_name_oh_inputs)
    FingerPrint_policy_Flat = tf.keras.layers.Flatten()(FingerPrint_policy_inputs)
    FingerPrint_eps_time_Flat = tf.keras.layers.Flatten()(FingerPrint_eps_time_inputs)

    conv_layers = [(16, 2), (32, 2), (32, 2), (32, 2)]
    conv_out = inputs
    conv_out = tf.cast(conv_out, tf.float32) / 255.
    for i, (num_ch, num_blocks) in enumerate(conv_layers):
        conv_out = tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                          name=f'conv_{i}', kernel_initializer=ortho_init(1.0),
                                  bias_initializer=tf.keras.initializers.Constant(1.0))(conv_out)
        conv_out = tf.keras.layers.MaxPooling2D((3, 3), padding='same', strides=2, name=f'maxP_{i}')(conv_out)

        for j in range(num_blocks):
            with tf.name_scope('residual_%d_%d' % (i, j)):
                block_input = conv_out
                conv_out = tf.keras.layers.Activation(tf.nn.relu, name=f'act_{i}_{j}_1')(conv_out)
                conv_out = tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                                  activation=tf.nn.relu, name=f'conv_{i}_{j}_1',
                                                  kernel_initializer=ortho_init(1.0),
                                                  bias_initializer=tf.keras.initializers.Constant(1.0)
                                                  )(conv_out)
                conv_out = tf.keras.layers.Conv2D(filters=num_ch, kernel_size=3, strides=(1, 1), padding='same',
                                                  name=f'conv_{i}_{j}_2',  kernel_initializer=ortho_init(1.0),
                                                  bias_initializer=tf.keras.initializers.Constant(1.0))(conv_out)
                conv_out += block_input

    conv_out = tf.keras.layers.Activation(tf.nn.relu)(conv_out)
    conv_out = tf.keras.layers.BatchNormalization()(conv_out)
    conv_out = tf.keras.layers.Flatten()(conv_out)

    outputs = tf.keras.layers.concatenate([conv_out, FingerPrint_policy_Flat, FingerPrint_eps_time_Flat, agent_name_oh_flatted])

    return tf.keras.Model(inputs=[inputs, agent_name_oh_inputs, FingerPrint_policy_inputs, FingerPrint_eps_time_inputs],
                          outputs=outputs, name=model_name)

