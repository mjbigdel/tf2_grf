import tensorflow as tf


# Model
class Network(tf.Module):
    def __init__(self, config, agent_ids):
        super().__init__()
        self.config = config
        self.agent_ids = agent_ids

    def init_base_model(self):
        if self.config.network == 'cnn':
            return cnn(self.config)
        if self.config.network == 'mlp':
            return mlp(self.config)

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
            outputs_a = tf.keras.layers.Dense(units=self.config.num_actions, activation=None,
                                              kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                              bias_initializer=tf.keras.initializers.Constant(0.0),
                                              name=name)(head_inputs)
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
                action_head_a = tf.keras.layers.Dense(units=self.config.num_actions, activation=None,
                                                      kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                                      bias_initializer=tf.keras.initializers.Constant(0.0),
                                                      name='action_' + name)(head_inputs)

            with tf.name_scope(f'state_value_{agent_id}'):
                state_head_a = tf.keras.layers.Dense(units=1, activation=None,
                                                     kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                                     bias_initializer=tf.keras.initializers.Constant(0.0),
                                                     name='state_' + name)(head_inputs)

            action_scores_mean = tf.reduce_mean(action_head_a, 1)
            action_scores_centered = action_head_a - tf.expand_dims(action_scores_mean, 1)
            head = state_head_a + action_scores_centered

            head = tf.keras.Model(inputs=base_model.inputs, outputs=head)
            heads.append(head)

        return heads

    def build_models(self):
        if self.config.dueling:
            return self.build_models_and_dueling_agent_heads()
        else:
            return self.build_models_and_agent_heads()


def cnn(config):
    inputs = tf.keras.layers.Input(shape=config.obs_shape)
    conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=(4, 4),
                                   activation=tf.nn.relu, name='conv1')(inputs)
    conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=4, strides=(2, 2),
                                   activation=tf.nn.relu, name='conv2')(conv1)
    conv3 = tf.keras.layers.Conv2D(filters=16, kernel_size=2, strides=(1, 1),
                                   activation=tf.nn.relu, name='conv3')(conv2)
    flatten = tf.keras.layers.Flatten(name='flatten')(conv3)
    outputs = tf.keras.layers.Dense(config.fc1_dims, activation=tf.nn.relu, name='dense1')(flatten)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=config.network)


def mlp(config):
    inputs = tf.keras.layers.Input((config.obs_shape), name='Input_obs')
    outputs = tf.keras.layers.Flatten(name='flatten')(inputs)
    outputs = tf.keras.layers.Dense(64, name='h1_dense')(outputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=config.network)
