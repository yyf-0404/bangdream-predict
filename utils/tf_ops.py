# utils/tf_ops.py
import tensorflow as tf

def set_global_seed(seed: int):
    import os, random, numpy as np
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def safe_log1p(y):
    return tf.math.log1p(tf.maximum(y, 0.0))

def huber_per_elem(y_true, y_pred, delta=1.0):
    err = y_pred - y_true
    abs_err = tf.abs(err)
    quad = tf.minimum(abs_err, delta)
    lin = abs_err - quad
    return 0.5 * tf.square(quad) + delta * lin

def masked_mean(x, mask, axis=None, eps=1e-8):
    mask_f = tf.cast(mask, tf.float32)
    num = tf.reduce_sum(x * mask_f, axis=axis)
    den = tf.reduce_sum(mask_f, axis=axis)
    return num / (den + eps)
def make_positions_tf(L: tf.Tensor, B: tf.Tensor) -> tf.Tensor:
    pos = tf.cast(tf.range(1, L + 1), tf.float32)[tf.newaxis, :]
    return tf.tile(pos, [B, 1])