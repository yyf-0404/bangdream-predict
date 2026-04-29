# model/cdf_shape.py
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from utils.tf_ops import safe_log1p, huber_per_elem, masked_mean
from model.layers.normalization import Log1pStandardize, MinMaxNormalizeClip
from model.layers.seasonal import SeasonHourSegNet, WeekHolCorrectionNet
from data.constants import HOUR_BINS, WEEK_BINS, HOL_BINS, NUM_EVENT_TYPES, NUM_BANDS, NUM_TIERS, NUM_SERVERS

class CDFShapeMLP(keras.Model):
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

        # 标准化层
        self.norm_y = Log1pStandardize(y_mean, y_std, name="norm_y")
        self.norm_ctx = MinMaxNormalizeClip(ctx_min, ctx_max, name="norm_ctx")

        # 上下文嵌入
        self.e_event = layers.Embedding(NUM_EVENT_TYPES, 8, name="emb_event")
        self.e_band  = layers.Embedding(NUM_BANDS, 8, name="emb_band")
        self.e_tier  = layers.Embedding(NUM_TIERS, 8, name="emb_tier")
        self.e_server= layers.Embedding(NUM_SERVERS, 4, name="emb_server")
        self.ctx_dense = layers.Dense(64, activation="relu")

        # 历史编码器 (GRU)
        self.hist_gru = layers.GRU(hidden, return_sequences=False, dropout=0,
                                    name="hist_gru")
        self.hist_proj = layers.Dense(hidden, activation="relu")

        # 形状头：输出 K 段 logits
        self.shape_head = keras.Sequential([
            layers.Dense(hidden, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(self._K_segments_cfg),
        ], name="shape_head")

        # 季节网络
        self.season_hour = SeasonHourSegNet(K_segments=K_segments, hour_bins=HOUR_BINS, name="season_hour")
        self.weekhol_corr = WeekHolCorrectionNet(week_bins=WEEK_BINS, hol_bins=HOL_BINS, name="weekhol_corr")

        # 指标
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.sum_loss_tracker = keras.metrics.Mean(name="sum_loss")
        self.shape_loss_tracker = keras.metrics.Mean(name="shape_loss")

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
        x_hist = tf.cast(inputs["x_hist"], tf.float32)          # (B, Lh, 1)
        hist_len = tf.cast(inputs["hist_len"], tf.int32)        # (B,)
        future_len = tf.cast(inputs["future_len"], tf.int32)    # (B,)
        hh = tf.cast(inputs["exo_hist_hour"], tf.int32)
        hw = tf.cast(inputs["exo_hist_week"], tf.int32)
        hhol= tf.cast(inputs["exo_hist_hol"], tf.int32)
        fh = tf.cast(inputs["future_exo_hour"], tf.int32)
        fw = tf.cast(inputs["future_exo_week"], tf.int32)
        fhol = tf.cast(inputs["future_exo_hol"], tf.int32)

        B = tf.shape(x_hist)[0]
        Lh = tf.shape(x_hist)[1]
        Lf = tf.shape(fh)[1]

        hist_mask = tf.sequence_mask(hist_len, maxlen=Lh)
        future_mask = tf.sequence_mask(future_len, maxlen=Lf)
        hist_mask3 = hist_mask[..., tf.newaxis]
        future_mask3 = future_mask[..., tf.newaxis]
        hist_mask_f = tf.cast(hist_mask3, tf.float32)

        # 进度分段
        th = tf.cast(hist_len, tf.float32)
        tf_len = tf.cast(future_len, tf.float32)
        T = tf.maximum(th + tf_len, 1.0)

        pos_h = tf.cast(tf.range(Lh), tf.float32)[tf.newaxis, :] / T[:, tf.newaxis]
        pos_f = (th[:, tf.newaxis] + tf.cast(tf.range(Lf), tf.float32)[tf.newaxis, :]) / T[:, tf.newaxis]

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

        # 历史编码
        hist_feat = tf.concat([x_hist_norm, pos_h[..., tf.newaxis]], axis=-1)
        h_vec = self.hist_gru(hist_feat, mask=hist_mask, training=training)
        h_vec = self.hist_proj(h_vec)

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

        # 形状分布 u
        gate_in = tf.concat([h_vec, ctx_vec], axis=-1)
        u_logits = self.shape_head(gate_in, training=training)   # (B, K)
        u = tf.nn.softmax(u_logits, axis=-1)

        # CDF 推断总量
        # 先对 x_hist_ds 裁剪，再转回 base 域求和
        x_hist_ds_clipped = tf.clip_by_value(x_hist_ds, -5.0, 15.0)  # 与 pred_log 裁剪一致
        x_hist_base = tf.math.expm1(x_hist_ds_clipped) * hist_mask_f
        prefix_sum_base = tf.reduce_sum(x_hist_base, axis=[1,2])
        p0 = th / T
        upos = p0 * K
        j = tf.cast(tf.floor(upos), tf.int32)
        j = tf.clip_by_value(j, 0, self._K_segments_cfg - 1)
        frac = tf.clip_by_value(upos - tf.cast(j, tf.float32), 0.0, 1.0)

        cumsum_u = tf.cumsum(u, axis=-1)
        prev = tf.gather(cumsum_u, tf.maximum(j - 1, 0), batch_dims=1)
        prev = tf.where(j > 0, prev, tf.zeros_like(prev))
        uj = tf.gather(u, j, batch_dims=1)
        cdf_p0 = prev + frac * uj
        cdf_p0 = tf.clip_by_value(cdf_p0, 1e-4, 1.0 - 1e-6)
        total_hat_base = prefix_sum_base / cdf_p0

        # 分配未来量
        one_hot_j = tf.one_hot(j, depth=self._K_segments_cfg, dtype=tf.float32)
        after_j = tf.cast(tf.range(self._K_segments_cfg)[tf.newaxis, :] > j[:, tf.newaxis], tf.float32)
        tail_frac_in_j = (1.0 - frac)[:, tf.newaxis]
        tail_mass = u * after_j + u * one_hot_j * tail_frac_in_j

        seg_onehot = tf.one_hot(seg_future, depth=self._K_segments_cfg, dtype=tf.float32)
        seg_onehot = seg_onehot * tf.cast(future_mask[..., tf.newaxis], tf.float32)
        n_per_seg = tf.reduce_sum(seg_onehot, axis=1) + 1e-6
        per_step_mass = tf.reduce_sum(seg_onehot * (tail_mass / n_per_seg)[:, tf.newaxis, :], axis=-1)
        per_step_mass = per_step_mass * tf.cast(future_mask, tf.float32)

        y_ds_base = total_hat_base[:, tf.newaxis] * per_step_mass
        y_ds_base = tf.maximum(y_ds_base, 0.0)

        # 最终预测
        log_y_ds = tf.math.log1p(y_ds_base)[..., tf.newaxis]
        log_y_hat = log_y_ds + s_fut + corr_fut
        log_y_hat = tf.clip_by_value(log_y_hat, -5.0, 15.0)
        y_hat = tf.math.expm1(log_y_hat) * tf.cast(future_mask3, tf.float32)

        return {
            "y_hat": y_hat,
            "future_mask": future_mask3,
            "pred_log": log_y_hat,
            "s_future": s_fut,
            "corr_future": corr_fut,
            "u": u,
            "cdf_p0": cdf_p0,
            "total_hat_base": total_hat_base,
            "prefix_sum_base": prefix_sum_base,
            "y_ds_base": y_ds_base,
        }

    def compute_loss(self, y_true, out):
        y_true = tf.cast(y_true, tf.float32)
        future_mask = out["future_mask"]
        mask_f = tf.cast(future_mask, tf.float32)

        log_true = safe_log1p(y_true)
        log_pred = out["pred_log"]
        # 点级损失
        per_elem = huber_per_elem(log_true, log_pred, delta=1.0)
        loss_time = tf.reduce_sum(per_elem * mask_f) / (tf.reduce_sum(mask_f) + 1e-8)

        # 总量损失
        sum_true = tf.reduce_sum(y_true * mask_f, axis=[1,2])
        sum_pred = tf.reduce_sum(out["y_hat"] * mask_f, axis=[1,2])
        loss_sum = tf.reduce_mean(huber_per_elem(safe_log1p(sum_true), safe_log1p(sum_pred), delta=1.0))

        # 形状损失 (CDF)
        y_true_ds = tf.maximum(y_true - (tf.math.expm1(out["s_future"] + out["corr_future"])), 0.0) * mask_f
        y_pred_ds = out["y_ds_base"][..., tf.newaxis] * mask_f

        c_true = tf.cumsum(y_true_ds, axis=1)
        c_pred = tf.cumsum(y_pred_ds, axis=1)
        tot_true = tf.reduce_sum(y_true_ds, axis=1, keepdims=True) + 1e-6
        tot_pred = tf.reduce_sum(y_pred_ds, axis=1, keepdims=True) + 1e-6
        cdf_true = c_true / tot_true
        cdf_pred = c_pred / tot_pred
        l1 = tf.abs(cdf_true - cdf_pred)
        loss_shape = tf.reduce_sum(l1 * mask_f) / (tf.reduce_sum(mask_f) + 1e-8)

        loss = loss_time + 10.0 * loss_sum + 10.0 * loss_shape
        self.loss_tracker.update_state(loss)
        self.sum_loss_tracker.update_state(loss_sum)
        self.shape_loss_tracker.update_state(loss_shape)
        return loss

    def train_step(self, data):
        x, y_true, sw = data
        with tf.GradientTape() as tape:
            out = self(x, training=True)
            loss = self.compute_loss(y_true, out)
        grads = tape.gradient(loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result(), "sum_loss": self.sum_loss_tracker.result(),
                "shape_loss": self.shape_loss_tracker.result()}

    def test_step(self, data):
        x, y_true, sw = data
        out = self(x, training=False)
        loss = self.compute_loss(y_true, out)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.sum_loss_tracker, self.shape_loss_tracker]