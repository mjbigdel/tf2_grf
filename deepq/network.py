import tensorflow as tf


# Model
class Network(tf.Module):
    def __init__(self, config, agent_ids):
        super().__init__()
        self.config = config
        self.agent_ids = agent_ids

    def init_base_model(self):
        inputs = tf.keras.layers.Input((self.config.obs_shape), name='Input_obs')
        outputs = tf.keras.layers.Flatten(name='flatten')(inputs)
        outputs = tf.keras.layers.Dense(64, name='h1_dense')(outputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name=self.config.network)

    def build_model(self, base_model):
        heads = []
        for agent_id in self.agent_ids:
            outputs_a = tf.keras.layers.Dense(self.config.num_actions,
                                              name=f'q_values_{agent_id}')(base_model.layers[-1].output)
            heads.append(tf.keras.Model(inputs=base_model.inputs, outputs=outputs_a,
                                        name=self.config.network + f'_agent_{agent_id}'))
        return heads
