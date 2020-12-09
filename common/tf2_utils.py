import tensorflow as tf

@tf.function
def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.math.abs(x) < delta,
        tf.math.square(x) * 0.5,
        delta * (tf.math.abs(x) - 0.5 * delta)
    )
