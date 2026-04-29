# model/layers/normalization.py
import numpy as np
import tensorflow as tf
import keras
from keras import layers

@keras.saving.register_keras_serializable(package="BangDream")
class Log1pStandardize(layers.Layer):
    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.mean_value = np.asarray(mean, dtype=np.float32)
        self.std_value = np.asarray(std, dtype=np.float32)

    def build(self, _):
        self.mean = self.add_weight(
            name="mean", shape=self.mean_value.shape, dtype="float32",
            initializer=keras.initializers.Constant(self.mean_value), trainable=False
        )
        self.std = self.add_weight(
            name="std", shape=self.std_value.shape, dtype="float32",
            initializer=keras.initializers.Constant(self.std_value), trainable=False
        )

    def call(self, y):
        from utils.tf_ops import safe_log1p   # 避免循环导入
        y_log = safe_log1p(y)
        return (y_log - self.mean) / (self.std + 1e-8)

@keras.saving.register_keras_serializable(package="BangDream")
class MinMaxNormalizeClip(layers.Layer):
    def __init__(self, min_val, max_val, clip_low=0.0, clip_high=1.5, **kwargs):
        super().__init__(**kwargs)
        self.min_init = float(min_val)
        self.max_init = float(max_val)
        self.clip_low = float(clip_low)
        self.clip_high = float(clip_high)

    def build(self, _):
        self.min_val = self.add_weight(
            name="min_val", shape=(), dtype="float32",
            initializer=keras.initializers.Constant(self.min_init), trainable=False
        )
        self.max_val = self.add_weight(
            name="max_val", shape=(), dtype="float32",
            initializer=keras.initializers.Constant(self.max_init), trainable=False
        )

    def call(self, x):
        denom = tf.maximum(self.max_val - self.min_val, 1e-8)
        x_norm = (tf.cast(x, tf.float32) - self.min_val) / denom
        return tf.clip_by_value(x_norm, self.clip_low, self.clip_high)