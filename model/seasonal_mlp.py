# model/seasonal_mlp.py
import numpy as np
import tensorflow as tf
import keras
from keras import layers

from utils.tf_ops import safe_log1p, huber_per_elem
from model.layers.normalization import Log1pStandardize, MinMaxNormalizeClip
from model.layers.seasonal import SeasonHourSegNet, WeekHolCorrectionNet
from data.constants import HOUR_BINS, WEEK_BINS, HOL_BINS, NUM_EVENT_TYPES, NUM_BANDS, NUM_TIERS, NUM_SERVERS

class SeasonalMLP(keras.Model):
    def __init__(self, y_mean, y_std, ctx_min, ctx_max,
                 hidden=128, dropout=0.1, K_segments=96, **kwargs):
        super().__init__(**kwargs)
        self._y_mean_cfg = np.asarray(y_mean, dtype=np.float32).tolist()
        self._y_std_cfg  = np.asarray(y_std, dtype=np.float32).tolist()
        self._ctx_min_cfg = float(ctx_min)
        self._ctx_max_cfg = float(ctx_max)
        self._hidden_cfg = int(hidden)
        self._dropout_cfg = float(dropout)
        self._K_segments_cfg = int(K_segments)
        self._hour_bins_cfg = HOUR_BINS

        self.norm_y = Log1pStandardize(y_mean, y_std, name="norm_y")
        self.norm_ctx = MinMaxNormalizeClip(ctx_min, ctx_max, name="norm_ctx")

        # 上下文嵌入
        self.e_event = layers.Embedding(NUM_EVENT_TYPES, 8, name="emb_event")
        self.e_band  = layers.Embedding(NUM_BANDS, 8, name="emb_band")
        self.e_tier  = layers.Embedding(NUM_TIERS, 8, name="emb_tier")
        self.e_server= layers.Embedding(NUM_SERVERS, 4, name="emb_server")
        self.ctx_dense = layers.Dense(64, activation="relu")

        # 未来外生嵌入
        self.e_fhour = layers.Embedding(HOUR_BINS + 1, 4, name="emb_fhour")
        self.e_fweek = layers.Embedding(WEEK_BINS + 1, 4, name="emb_fweek")
        self.e_fhol  = layers.Embedding(HOL_BINS + 1, 2, name="emb_fhol")

        # 主干 MLP
        self.mlp = keras.Sequential([
            layers.Dense(hidden, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(hidden, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(1),
        ], name="trunk_mlp")

        # 季节网络
        self.season_hour = SeasonHourSegNet(
            K_segments=K_segments, hour_bins=HOUR_BINS, name="season_hour"
        )
        self.weekhol_corr = WeekHolCorrectionNet(
            week_bins=WEEK_BINS, hol_bins=HOL_BINS, name="weekhol_corr"
        )

        self.loss_tracker = keras.metrics.Mean(name="loss")

    def get_config(self):
        config = super().get_config()
        config.update({
            "y_mean": self._y_mean_cfg,
            "y_std": self._y_std_cfg,
            "ctx_min": self._ctx_min_cfg,
            "ctx_max": self._ctx_max_cfg,
            "hidden": self._hidden_cfg,
            "dropout": self._dropout_cfg,
            "K_segments": self._K_segments_cfg,
            "hour_bins": self._hour_bins_cfg,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["y_mean"] = np.asarray(config["y_mean"], dtype=np.float32)
        config["y_std"] = np.asarray(config["y_std"], dtype=np.float32)
        return cls(**config)

    def call(self, inputs, training=False):
        x_hist = tf.cast(inputs["x_hist"], tf.float32)
        hist_len = tf.cast(inputs["hist_len"], tf.int32)
        future_len = tf.cast(inputs["future_len"], tf.int32)

        hh = tf.cast(inputs["exo_hist_hour"], tf.int32)
        hw = tf.cast(inputs["exo_hist_week"], tf.int32)
        hhol = tf.cast(inputs["exo_hist_hol"], tf.int32)

        fh = tf.cast(inputs["future_exo_hour"], tf.int32)
        fw = tf.cast(inputs["future_exo_week"], tf.int32)
        fhol = tf.cast(inputs["future_exo_hol"], tf.int32)

        Lh = tf.shape(x_hist)[1]
        Lf = tf.shape(fh)[1]

        hist_mask = tf.sequence_mask(hist_len, maxlen=Lh)
        future_mask = tf.sequence_mask(future_len, maxlen=Lf)
        hist_mask3 = hist_mask[..., tf.newaxis]
        future_mask3 = future_mask[..., tf.newaxis]

        # 进度分段
        B = tf.shape(x_hist)[0]
        th = tf.cast(hist_len, tf.float32)
        tf_len = tf.cast(future_len, tf.float32)
        T_total = tf.maximum(th + tf_len, 1.0)

        pos_h = tf.cast(tf.range(Lh), tf.float32)[tf.newaxis, :] / T_total[:, tf.newaxis]
        pos_f = (th[:, tf.newaxis] + tf.cast(tf.range(Lf), tf.float32)[tf.newaxis, :]) / T_total[:, tf.newaxis]

        K = float(self._K_segments_cfg)
        seg_hist = tf.cast(tf.floor(pos_h * K), tf.int32)
        seg_hist = tf.clip_by_value(seg_hist, 0, self._K_segments_cfg - 1)
        seg_future = tf.cast(tf.floor(pos_f * K), tf.int32)
        seg_future = tf.clip_by_value(seg_future, 0, self._K_segments_cfg - 1)

        # 季节效应
        s_hist = self.season_hour(hh, hw, hhol, seg_hist, hist_mask)
        s_fut  = self.season_hour(fh, fw, fhol, seg_future, future_mask)
        corr_hist = self.weekhol_corr(hw, hhol, hist_mask)
        corr_fut  = self.weekhol_corr(fw, fhol, future_mask)

        # 去季节化历史
        x_hist_log = safe_log1p(x_hist)
        x_hist_ds = x_hist_log - s_hist - corr_hist
        if not self.norm_y.built:
            self.norm_y.build(x_hist.shape)
        x_hist_norm = (x_hist_ds - self.norm_y.mean) / (self.norm_y.std + 1e-8)

        # 池化
        hist_mask_f = tf.cast(hist_mask3, tf.float32)
        hist_mean = tf.reduce_sum(x_hist_norm * hist_mask_f, axis=1) / (
            tf.reduce_sum(hist_mask_f, axis=1) + 1e-8)
        idx = hist_len - 1
        hist_last = tf.gather(x_hist_norm[:, :, 0], idx, batch_dims=1)[:, tf.newaxis]

        # 上下文
        ctx_cat = tf.cast(inputs["context_cat"], tf.int32)
        e0 = self.e_event(ctx_cat[:, 0])
        e1 = self.e_band(ctx_cat[:, 1])
        e2 = self.e_tier(ctx_cat[:, 2])
        e3 = self.e_server(ctx_cat[:, 3])
        ctx_num = tf.cast(inputs["context_num"], tf.float32)[:, tf.newaxis]
        ctx_num = self.norm_ctx(ctx_num)
        ctx_vec = tf.concat([e0, e1, e2, e3, ctx_num], axis=-1)
        ctx_vec = self.ctx_dense(ctx_vec)

        # 未来步特征
        global_feat = tf.concat([hist_mean, hist_last, ctx_vec], axis=-1)
        eh = self.e_fhour(fh)
        ew = self.e_fweek(fw)
        ehol = self.e_fhol(fhol)
        pos_f_norm = tf.cast(tf.range(Lf), tf.float32)[tf.newaxis, :, tf.newaxis] / tf.cast(Lf, tf.float32)
        pos_f_tile = tf.tile(pos_f_norm, [B, 1, 1])
        future_step_feat = tf.concat([eh, ew, ehol, pos_f_tile], axis=-1)

        global_repeat = tf.tile(global_feat[:, tf.newaxis, :], [1, Lf, 1])
        step_input = tf.concat([global_repeat, future_step_feat], axis=-1)

        # 预测去季节化的 log 值
        pred_log_base = self.mlp(step_input, training=training)
        pred_log_base = pred_log_base * tf.cast(future_mask3, tf.float32)

        # 加上季节效应 -> 最终 log 预测
        pred_log = pred_log_base + s_fut + corr_fut
        pred_log = tf.clip_by_value(pred_log, -5.0, 15.0)

        y_hat = tf.math.expm1(pred_log) * tf.cast(future_mask3, tf.float32)

        return {
            "y_hat": y_hat,
            "future_mask": future_mask3,
            "pred_log": pred_log,
            "pred_log_base": pred_log_base,
            "s_future": s_fut,
            "corr_future": corr_fut,
        }

    def compute_loss(self, y_true, out):
        y_true = tf.cast(y_true, tf.float32)
        future_mask = out["future_mask"]
        mask_f = tf.cast(future_mask, tf.float32)

        log_true = safe_log1p(y_true)
        log_pred = out["pred_log"]

        per_elem = huber_per_elem(log_true, log_pred, delta=1.0)
        loss = tf.reduce_sum(per_elem * mask_f) / (tf.reduce_sum(mask_f) + 1e-8)
        return loss

    def train_step(self, data):
        x, y_true, sw = data
        with tf.GradientTape() as tape:
            out = self(x, training=True)
            loss = self.compute_loss(y_true, out)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y_true, sw = data
        out = self(x, training=False)
        loss = self.compute_loss(y_true, out)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]