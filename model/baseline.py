# model/baseline.py
import tensorflow as tf
import numpy as np
import keras
from keras import layers
from utils.tf_ops import safe_log1p, huber_per_elem
from model.layers.normalization import Log1pStandardize, MinMaxNormalizeClip
from data.constants import HOUR_BINS, WEEK_BINS, HOL_BINS, NUM_EVENT_TYPES, NUM_BANDS, NUM_TIERS, NUM_SERVERS

class SimpleMLPBaseline(keras.Model):
    def __init__(self, y_mean, y_std, ctx_min, ctx_max, hidden=128, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self._y_mean_cfg = np.asarray(y_mean, dtype=np.float32).tolist()
        self._y_std_cfg  = np.asarray(y_std, dtype=np.float32).tolist()
        self._ctx_min_cfg = float(ctx_min)
        self._ctx_max_cfg = float(ctx_max)
        self._hidden_cfg = int(hidden)
        self._dropout_cfg = float(dropout)

        self.norm_y = Log1pStandardize(y_mean, y_std, name="norm_y")
        self.norm_ctx = MinMaxNormalizeClip(ctx_min, ctx_max, name="norm_ctx")   # 新增

        # 上下文嵌入
        self.e_event = layers.Embedding(NUM_EVENT_TYPES, 8, name="emb_event")
        self.e_band  = layers.Embedding(NUM_BANDS, 8, name="emb_band")
        self.e_tier  = layers.Embedding(NUM_TIERS, 8, name="emb_tier")
        self.e_server= layers.Embedding(NUM_SERVERS, 4, name="emb_server")
        self.ctx_dense = layers.Dense(64, activation="relu")

        # 未来外生变量嵌入（注意索引从1开始，0留作padding）
        self.e_fhour = layers.Embedding(HOUR_BINS + 1, 4, name="emb_fhour")   # 0~48，0为padding
        self.e_fweek = layers.Embedding(WEEK_BINS + 1, 4, name="emb_fweek")    # 0~7
        self.e_fhol  = layers.Embedding(HOL_BINS + 1, 2, name="emb_fhol")     # 0~2

        # 主干 MLP：输入 = [全局特征, 未来外生嵌入, 未来位置]
        self.mlp = keras.Sequential([
            layers.Dense(hidden, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(hidden, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(1),
        ], name="trunk_mlp")

        self.loss_tracker = keras.metrics.Mean(name="loss")
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "y_mean": self._y_mean_cfg,
            "y_std": self._y_std_cfg,
            "hidden": self._hidden_cfg,
            "dropout": self._dropout_cfg,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # 如果 y_mean 等被存为 list，需要转回 numpy
        config = config.copy()
        if "y_mean" in config:
            config["y_mean"] = np.asarray(config["y_mean"], dtype=np.float32)
        if "y_std" in config:
            config["y_std"] = np.asarray(config["y_std"], dtype=np.float32)
        return cls(**config)
    def call(self, inputs, training=False):
        x_hist = tf.cast(inputs["x_hist"], tf.float32)
        hist_len = tf.cast(inputs["hist_len"], tf.int32)
        future_len = tf.cast(inputs["future_len"], tf.int32)

        # ---- 上下文 ----
        ctx_cat = tf.cast(inputs["context_cat"], tf.int32)
        e0 = self.e_event(ctx_cat[:, 0])
        e1 = self.e_band(ctx_cat[:, 1])
        e2 = self.e_tier(ctx_cat[:, 2])
        e3 = self.e_server(ctx_cat[:, 3])
        ctx_num = tf.cast(inputs["context_num"], tf.float32)[:, tf.newaxis]
        ctx_num = self.norm_ctx(ctx_num)          # 归一化到 0~1.5
        ctx_vec = tf.concat([e0, e1, e2, e3, ctx_num], axis=-1)
        ctx_vec = self.ctx_dense(ctx_vec)                     # (B, 64)

        # ---- 历史特征（简单池化） ----
        # 归一化历史
        x_hist_log = safe_log1p(x_hist)                       # (B, Lh, 1)
        if not self.norm_y.built:
            self.norm_y.build(x_hist.shape)
        x_hist_norm = (x_hist_log - self.norm_y.mean) / (self.norm_y.std + 1e-8)
        # 历史掩码
        hist_mask = tf.sequence_mask(hist_len, maxlen=tf.shape(x_hist)[1], dtype=tf.float32)
        hist_mean = tf.reduce_sum(
            x_hist_norm * hist_mask[:, :, tf.newaxis], axis=1
        ) / (tf.reduce_sum(hist_mask, axis=1, keepdims=True) + 1e-8)  # (B, 1)
        # 最后一步
        idx = hist_len - 1
        hist_last = tf.gather(x_hist_norm[:, :, 0], idx, batch_dims=1)[:, tf.newaxis]  # (B, 1)

        # ---- 全局特征向量 ----
        global_feat = tf.concat([hist_mean, hist_last, ctx_vec], axis=-1)  # (B, 2+64)

        # ---- 未来外生变量 ---- 
        fh = tf.cast(inputs["future_exo_hour"], tf.int32)    # (B, Lf) 值域 0~48
        fw = tf.cast(inputs["future_exo_week"], tf.int32)    # 0~7
        fhol = tf.cast(inputs["future_exo_hol"], tf.int32)   # 0~2
        Lf = tf.shape(fh)[1]

        eh = self.e_fhour(fh)                                # (B, Lf, 4)
        ew = self.e_fweek(fw)
        ehol = self.e_fhol(fhol)

        # 未来位置编码（归一化到 0-1）
        pos_f = tf.cast(tf.range(Lf)[tf.newaxis, :, tf.newaxis], tf.float32) / tf.cast(Lf, tf.float32)
        pos_f = tf.tile(pos_f, [tf.shape(fh)[0], 1, 1])      # (B, Lf, 1)

        # 拼接未来步特征
        future_step_feat = tf.concat([eh, ew, ehol, pos_f], axis=-1)  # (B, Lf, 4+4+2+1)

        # 将全局特征复制到每个未来步
        global_repeat = tf.tile(global_feat[:, tf.newaxis, :], [1, Lf, 1])  # (B, Lf, 66)

        # 每步的完整输入
        step_input = tf.concat([global_repeat, future_step_feat], axis=-1)

        pred_log = self.mlp(step_input, training=training)  # (B, Lf, 1)，值域不受限

        # 掩码处理
        future_mask = tf.sequence_mask(future_len, maxlen=tf.shape(pred_log)[1],
                                    dtype=tf.float32)[:, :, tf.newaxis]
        # 不做任何非线性，直接输出 log1p 空间的预测
        pred_log = pred_log * future_mask
        pred_log = tf.clip_by_value(pred_log, -5.0, 15.0)   # log(y) 合理范围

        # 如果需要真实值用于可视化，可以转换，但损失直接用 pred_log 与 log1p(y_true) 比较
        y_hat = tf.math.expm1(pred_log)  # 可选项，不用于损失
        return {
            "y_hat": y_hat,            # 用于最终可视化
            "future_mask": future_mask,
            "pred_log": pred_log       # 用于损失计算
        }
    def compute_loss(self, y_true, out):
        y_true = tf.cast(y_true, tf.float32)
        future_mask = out["future_mask"]
        mask_f = tf.cast(future_mask, tf.float32)

        log_true = safe_log1p(y_true)                 # (B, Lf, 1)
        log_pred = out["pred_log"]                    # 直接使用 log1p 空间的预测

        per_elem = huber_per_elem(log_true, log_pred, delta=1.0)
        loss = tf.reduce_sum(per_elem * mask_f) / (tf.reduce_sum(mask_f) + 1e-8)
        return loss

    def train_step(self, data):
        x, y_true, sample_weight = data
        with tf.GradientTape() as tape:
            out = self(x, training=True)
            loss = self.compute_loss(y_true, out)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y_true, sample_weight = data
        out = self(x, training=False)
        loss = self.compute_loss(y_true, out)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]