import tensorflow as tf

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
    outputs = tf.keras.layers.Dense(num_actions, name='outputs')(dense1)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)


def build_mlp(input_shape, num_actions, fc1_dims, fc2_dims, model_name):
    inputs = tf.keras.layers.Input(shape=input_shape)
    flatten = tf.keras.layers.Flatten(name='flatten')(inputs)
    dense1 = tf.keras.layers.Dense(fc1_dims, activation=tf.nn.relu, name='dense1')(flatten)
    dense2 = tf.keras.layers.Dense(fc2_dims, activation=tf.nn.relu, name='dense2')(dense1)
    outputs = tf.keras.layers.Dense(num_actions, name='outputs')(dense2)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)


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

    outputs = tf.keras.layers.Dense(num_actions, name='output')(dense2)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)


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
            outputs[agent_name] = tf.keras.layers.Dense(num_actions, name=f'output_{agent_name}')(dense2)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)


def build_ma_cnn(input_shape, num_actions, agents_names, fc1_dims, fc2_dims, share_weights, model_name):
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

        for agent_name in agents_names:
            outputs[agent_name] = tf.keras.layers.Dense(num_actions, name=f'output_{agent_name}')(dense1)

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
            outputs[agent_name] = tf.keras.layers.Dense(num_actions, name=f'output_{agent_name}')(dense1)

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
        lstm = tf.keras.layers.LSTM(units=rnn_units, dropout=dropout, return_sequences=False)(embed)
        dense2 = tf.keras.layers.Dense(fc2_dims, activation=tf.nn.relu, name='dense2')(lstm)

        for agent_name in agents_names:
            outputs[agent_name] = tf.keras.layers.Dense(num_actions, name=f'output_{agent_name}')(dense2)

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
            embed = tf.keras.layers.Embedding(input_dim=fc1_dims, output_dim=embed_dim, name=f'embed_{agent_name}')(dense1)
            lstm = tf.keras.layers.LSTM(units=rnn_units, dropout=dropout, return_sequences=False)(embed)
            dense2 = tf.keras.layers.Dense(fc2_dims, activation=tf.nn.relu, name=f'dense2_{agent_name}')(lstm)
            outputs[agent_name] = tf.keras.layers.Dense(num_actions, name=f'output_{agent_name}')(dense2)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)

