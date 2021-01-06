# import numpy as np
# import tensorflow as tf
# from tensorflow.python.keras.layers import LSTM
#
# """
# initializers
# """
# DEFAULT_SCALE = np.sqrt(2)
# DEFAULT_MODE = 'fan_in'
#
# def ortho_init(scale=DEFAULT_SCALE, mode=None):
#     def _ortho_init(shape, dtype, partition_info=None):
#         # lasagne ortho init for tf
#         shape = tuple(shape)
#         if len(shape) == 2: # fc: in, out
#             flat_shape = shape
#         elif (len(shape) == 3) or (len(shape) == 4): # 1d/2dcnn: (in_h), in_w, in_c, out
#             flat_shape = (np.prod(shape[:-1]), shape[-1])
#         a = np.random.standard_normal(flat_shape)
#         u, _, v = np.linalg.svd(a, full_matrices=False)
#         q = u if u.shape == flat_shape else v # pick the one with the correct shape
#         q = q.reshape(shape)
#         return (scale * q).astype(np.float32)
#     return _ortho_init
#
#
# def norm_init(scale=DEFAULT_SCALE, mode=DEFAULT_MODE):
#     def _norm_init(shape, dtype, partition_info=None):
#         shape = tuple(shape)
#         if len(shape) == 2:
#             n_in = shape[0]
#         elif (len(shape) == 3) or (len(shape) == 4):
#             n_in = np.prod(shape[:-1])
#         a = np.random.standard_normal(shape)
#         if mode == 'fan_in':
#             n = n_in
#         elif mode == 'fan_out':
#             n = shape[-1]
#         elif mode == 'fan_avg':
#             n = 0.5 * (n_in + shape[-1])
#         return (scale * a / np.sqrt(n)).astype(np.float32)
#
# DEFAULT_METHOD = ortho_init
# """
# layers
# """
# def conv(x, scope, n_out, f_size, stride=1, pad='VALID', f_size_w=None, act=tf.nn.relu,
#          conv_dim=1, init_scale=DEFAULT_SCALE, init_mode=None, init_method=DEFAULT_METHOD):
#     with tf.variable_scope(scope):
#         b = tf.get_variable("b", [n_out], initializer=tf.constant_initializer(0.0))
#         if conv_dim == 1:
#             n_c = x.shape[2].value
#             w = tf.get_variable("w", [f_size, n_c, n_out],
#                                 initializer=init_method(init_scale, init_mode))
#             z = tf.nn.conv1d(x, w, stride=stride, padding=pad) + b
#         elif conv_dim == 2:
#             n_c = x.shape[3].value
#             if f_size_w is None:
#                 f_size_w = f_size
#             w = tf.get_variable("w", [f_size, f_size_w, n_c, n_out],
#                                 initializer=init_method(init_scale, init_mode))
#             z = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=pad) + b
#         return act(z)
#
#
# def fc(x, scope, n_out, act=tf.nn.relu, init_scale=DEFAULT_SCALE,
#        init_mode=DEFAULT_MODE, init_method=DEFAULT_METHOD):
#     with tf.variable_scope(scope):
#         n_in = x.shape[1].value
#         w = tf.get_variable("w", [n_in, n_out],
#                             initializer=init_method(init_scale, init_mode))
#         b = tf.get_variable("b", [n_out], initializer=tf.constant_initializer(0.0))
#         z = tf.matmul(x, w) + b
#         return act(z)
#
#
# def batch_to_seq(x):
#     n_step = x.shape[0]
#     if len(x.shape) == 1:
#         x = tf.expand_dims(x, -1)
#     return tf.split(axis=0, num_or_size_splits=n_step, value=x)
#
#
# def seq_to_batch(x):
#     return tf.concat(x, axis=0)
#
#
# def lstm(xs, dones, s, scope, init_scale=DEFAULT_SCALE, init_mode=DEFAULT_MODE,
#          init_method=DEFAULT_METHOD):
#     xs = batch_to_seq(xs)
#     # need dones to reset states
#     dones = batch_to_seq(dones)
#     n_in = xs[0].shape[1].value
#     n_out = s.shape[0] // 2
#     with tf.variable_scope(scope):
#         wx = tf.get_variable("wx", [n_in, n_out*4],
#                              initializer=init_method(init_scale, init_mode))
#         wh = tf.get_variable("wh", [n_out, n_out*4],
#                              initializer=init_method(init_scale, init_mode))
#         b = tf.get_variable("b", [n_out*4], initializer=tf.constant_initializer(0.0))
#
#     s = tf.expand_dims(s, 0)
#     c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
#     for ind, (x, done) in enumerate(zip(xs, dones)):
#         c = c * (1-done)
#         h = h * (1-done)
#         z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
#         i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
#         i = tf.nn.sigmoid(i)
#         f = tf.nn.sigmoid(f)
#         o = tf.nn.sigmoid(o)
#         u = tf.tanh(u)
#         c = f*c + i*u
#         h = o*tf.tanh(c)
#         xs[ind] = h
#     s = tf.concat(axis=1, values=[c, h])
#     return seq_to_batch(xs), tf.squeeze(s)
#
#
# class LSTM_M(tf.keras.layers.Layer):
#     def __init__(self, units=32, scope='LSTM_M', init_scale=DEFAULT_SCALE, init_mode=DEFAULT_MODE, init_method=DEFAULT_METHOD):
#         super(LSTM_M, self).__init__()
#         print('init ----- ')
#         self.units = units
#         self.scope = scope
#         self.init_scale = init_scale
#         self.init_mode = init_mode
#         self.init_method = init_method
#
#
#     def build(self, input_shape):
#         n_in = input_shape[0][1]
#         # self.n_out = input_shape[0][0] // 2
#         self.n_out = self.units
#         self.states = np.zeros(self.n_out * 2, dtype=np.float32)
#         print('build ----- ')
#         print(f'n_in is {n_in}')
#         print(f'self.n_out is {self.n_out}')
#         with tf.name_scope(self.scope):
#             self.wx = self.add_weight(shape=(n_in, self.n_out * 4), initializer='uniform', name="wx")
#
#             self.wh = self.add_weight(shape=(self.n_out, self.n_out * 4), initializer='uniform', name="wh")
#
#             self.b = self.add_weight(shape=(self.n_out * 4), initializer=tf.constant_initializer(0.0), name="b")
#
#
#     def call(self, inputs, **kwargs):
#         print('call ----- ')
#         xs = batch_to_seq(inputs[0])
#         print(f'xs {xs}')
#         # need dones to reset states
#         dones = batch_to_seq(inputs[1])
#         print(f'dones {dones}')
#
#         s = tf.expand_dims(self.states, 0)
#         print(f's {s}')
#         c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
#         print(f'c {c}')
#         print(f'h {h}')
#         for ind, (x, done) in enumerate(zip(xs, dones)):
#             c = c * (1 - done)
#             h = h * (1 - done)
#             z = tf.matmul(x, self.wx)
#             z += tf.matmul(h, self.wh)
#             z += self.b
#             i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
#             i = tf.nn.sigmoid(i)
#             f = tf.nn.sigmoid(f)
#             o = tf.nn.sigmoid(o)
#             u = tf.tanh(u)
#             c = f * c + i * u
#             h = o * tf.tanh(c)
#             xs[ind] = h
#         s = tf.concat(axis=1, values=[c, h])
#         return seq_to_batch(xs), tf.squeeze(s)
#
#
# x = tf.ones((8, 16))
# dones = tf.ones((8, 1))
# # At instantiation, we don't know on what inputs this is going to get called
# lstm_layer = LSTM_M(32)
#
# # The layer's weights are created dynamically the first time the layer is called
# y = lstm_layer([x, dones])
# print(y)
#
# class Linear(tf.keras.layers.Layer):
#     def __init__(self, units=32):
#         super(Linear, self).__init__()
#         self.units = units
#
#     def build(self, input_shape):
#         self.w = self.add_weight(
#             shape=(input_shape[-1], self.units),
#             initializer="random_normal",
#             trainable=True,
#         )
#         self.b = self.add_weight(
#             shape=(self.units,), initializer="random_normal", trainable=True
#         )
#
#     def call(self, inputs):
#         return tf.matmul(inputs, self.w) + self.b
#
# # x = tf.ones((2, 2))
# # # At instantiation, we don't know on what inputs this is going to get called
# # linear_layer = Linear(32)
# #
# # # The layer's weights are created dynamically the first time the layer is called
# # y = linear_layer(x)
# # print(y)
#
#
#

import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)

# inputs_obs = tf.keras.layers.Input((4))
# outputs = tf.keras.layers.Dense(16)(inputs_obs)
# base_model = tf.keras.Model(inputs=inputs_obs, outputs=outputs)
#
#
# inputs_head_shape = base_model.output_shape[1:]
# print(inputs_head_shape)
# inputs_head = tf.keras.layers.Input((inputs_head_shape))
# outputs_a = tf.keras.layers.Dense(8)(inputs_head)
# model = tf.keras.Model(inputs=inputs_head, outputs=outputs_a)
#
# print(model.trainable_variables)


def init_base_network():
    inputs = tf.keras.layers.Input((4), name='Input_obs')
    outputs = tf.keras.layers.Dense(8, name='h1_dense')(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

base_model = init_base_network()
print(base_model)

def build_network_head():
    heads = []
    for agent in range(3):
        outputs_a = tf.keras.layers.Dense(4, name=f'q_values_{agent}')(base_model.layers[-1].output)
        heads.append(tf.keras.Model(inputs=base_model.inputs, outputs=outputs_a))
        heads[agent].summary()
    return heads


heads = build_network_head()

@tf.function
def compute_loss_a(agent):
    actions = [0, 1, 2, 3]
    obses = np.random.normal(0., 1., (3, 4, 4))
    rewards = [2., 3., 5., 1.]
    q_vals = heads[agent](obses[agent])
    one_hot = tf.one_hot(actions, 4)
    # print(one_hot)
    q_t_selected = tf.reduce_sum(q_vals * one_hot, 1)
    target_q_values = heads[agent](obses[agent] + np.random.normal(0., 1., obses[agent].shape))
    max_target_q_values = rewards + tf.reduce_max(target_q_values, axis=1)
    errors = q_t_selected - max_target_q_values
    return errors


@tf.function
def compute_loss():
    loss = []
    with tf.GradientTape() as tape:
        for agent in range(3):
            errors = compute_loss_a(agent)
            loss.append(tf.reduce_mean(errors))

        loss_all = tf.reduce_sum(loss)

    print(loss_all)
    param = tape.watched_variables()
    print(f'params: \n {param}')




compute_loss()