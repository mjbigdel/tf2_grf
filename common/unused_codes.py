
class CNN(tf.keras.Model, ABC):
    def __init__(self, num_actions):
        super(CNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=(4, 4), activation=tf.nn.relu)
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=(2, 2), activation=tf.nn.relu)
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.relu)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1024, activation=tf.nn.relu, name='cnn_dense')
        self.outputs = tf.keras.layers.Dense(num_actions)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        dense1 = self.dense1(x)
        return self.outputs(dense1)


class MLP(tf.keras.Model, ABC):
    def __init__(self, num_actions):
        super(MLP, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.outputs = tf.keras.layers.Dense(num_actions)

    def call(self, inputs, training=None, mask=None):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.outputs(x)


class MACNN(tf.keras.Model, ABC):
    def __init__(self, num_actions, num_agents, share_weights=False):
        super(MACNN, self).__init__()
        self.num_agents = num_agents
        if share_weights:
            self.outputs = {}
            self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=(4, 4), activation=tf.nn.relu)
            self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=(2, 2), activation=tf.nn.relu)
            self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.relu)
            self.flatten = tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense(1024, activation=tf.nn.relu, name='cnn_dense')

            for i in range(self.num_agents):
                self.outputs[i] = tf.keras.layers.Dense(num_actions)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        dense1 = self.dense1(x)
        outputs = {}
        for i in range(self.num_agents):
            outputs[i] = self.outputs[i](dense1)
        return outputs


class MAMLP(tf.keras.Model, ABC):
    def __init__(self, num_actions, num_agents, share_weights):
        super(MAMLP, self).__init__()
        self.num_agents = num_agents
        if share_weights:
            self.outputs = {}
            self.flatten = tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
            self.dense2 = tf.keras.layers.Dense(512, activation=tf.nn.relu)

            for i in range(self.num_agents):
                self.outputs[i] = tf.keras.layers.Dense(num_actions)

    def call(self, inputs, training=None, mask=None):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        outputs = {}
        for i in range(self.num_agents):
            outputs[i] = self.outputs[i](x)
        return outputs