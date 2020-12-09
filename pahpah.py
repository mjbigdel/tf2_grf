import tensorflow as tf
import numpy as np

"""
initializers
"""
DEFAULT_SCALE = np.sqrt(2)
DEFAULT_MODE = 'fan_in'

def ortho_init(scale=DEFAULT_SCALE, mode=None):
    def _ortho_init(shape, dtype, partition_info=None):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2: # fc: in, out
            flat_shape = shape
        elif (len(shape) == 3) or (len(shape) == 4): # 1d/2dcnn: (in_h), in_w, in_c, out
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        a = np.random.standard_normal(flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q).astype(np.float32)
    return _ortho_init


def norm_init(scale=DEFAULT_SCALE, mode=DEFAULT_MODE):
    def _norm_init(shape, dtype, partition_info=None):
        shape = tuple(shape)
        if len(shape) == 2:
            n_in = shape[0]
        elif (len(shape) == 3) or (len(shape) == 4):
            n_in = np.prod(shape[:-1])
        a = np.random.standard_normal(shape)
        if mode == 'fan_in':
            n = n_in
        elif mode == 'fan_out':
            n = shape[-1]
        elif mode == 'fan_avg':
            n = 0.5 * (n_in + shape[-1])
        return (scale * a / np.sqrt(n)).astype(np.float32)

DEFAULT_METHOD = ortho_init

def batch_to_seq(x):
    n_step = x.shape[0]
    if len(x.shape) == 1:
        x = tf.expand_dims(x, -1)
    return tf.split(axis=0, num_or_size_splits=n_step, value=x)


def seq_to_batch(x):
    return tf.concat(x, axis=0)


# def lstm(xs, dones, s, scope, init_scale=DEFAULT_SCALE, init_mode=DEFAULT_MODE,
#          init_method=DEFAULT_METHOD):
batch_size = 8
num_agents = 3
n_step = 16
fc_size = 128
init_scale= DEFAULT_SCALE
init_mode= DEFAULT_MODE
init_method= DEFAULT_METHOD
scope = 'test_lstm'

done_ = np.zeros((batch_size, n_step))
done_[-1] = 1
xs_ = np.ones((batch_size, n_step, fc_size))
# xs_ = batch_to_seq(xs_)
print(xs_.shape)
# need dones to reset states
# done_ = batch_to_seq(done_)
print(done_.shape)


s_ = [xs_, xs_, xs_]


xs = tf.keras.layers.Input(shape=(n_step, fc_size))
print(xs)
dones = tf.keras.layers.Input(shape=n_step)
print(dones)
s = tf.keras.layers.Input(shape=(num_agents, n_step, fc_size))
print(s)

n_in = xs[0].shape[1]
print(n_in)
n_out = s.shape[1] // 2
print(n_out)

def fn_():
    c = c * (1-done)
    h = h * (1-done)
    z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
    i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
    i = tf.nn.sigmoid(i)
    f = tf.nn.sigmoid(f)
    o = tf.nn.sigmoid(o)
    u = tf.tanh(u)
    c = f*c + i*u
    h = o*tf.tanh(c)
    return h

def fn_c(t):
    return tf.range(t, t + 3)

x = tf.map_fn(fn=lambda t: fn_c(t), elems=tf.constant([3, 5, 2]))

print(f'x = {x}')

with tf.name_scope(scope):
    wx = tf.Variable(initial_value=init_method(init_scale, init_mode)([n_out, n_out*4], xs_.dtype),
                     name="wx")
    wh = tf.Variable(initial_value=init_method(init_scale, init_mode)([n_out, n_out*4], xs_.dtype),
                     name="wh")
    b = tf.Variable(initial_value=tf.constant_initializer(0.0)([n_out*4], xs_.dtype), name="b")

s = tf.expand_dims(s, 0)
c, h = tf.split(axis=1, num_or_size_splits=2, value=s)

for ind, (x, done) in enumerate(zip(xs, dones)):
    c = c * (1-done)
    h = h * (1-done)
    z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
    i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
    i = tf.nn.sigmoid(i)
    f = tf.nn.sigmoid(f)
    o = tf.nn.sigmoid(o)
    u = tf.tanh(u)
    c = f*c + i*u
    h = o*tf.tanh(c)
    xs[ind] = h
s = tf.concat(axis=1, values=[c, h])
# return seq_to_batch(xs), tf.squeeze(s)
print(seq_to_batch(xs), tf.squeeze(s))