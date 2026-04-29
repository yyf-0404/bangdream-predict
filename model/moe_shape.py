# model/moe_shape.py
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from utils.tf_ops import safe_log1p, huber_per_elem, masked_mean
from model.layers.normalization import Log1pStandardize, MinMaxNormalizeClip
from model.layers.seasonal import SeasonHourSegNet, WeekHolCorrectionNet
from model.layers.moe import ShapeMoE
from data.constants import HOUR_BINS, WEEK_BINS, HOL_BINS, NUM_EVENT_TYPES, NUM_BANDS, NUM_TIERS, NUM_SERVERS

@keras.saving.register_keras_serializable(package="BangDream")
class MoeShapeModel(keras.Model):
    def __init__(self, y_mean, y_std, ctx_min, ctx_max,
                 hidden=128, dropout=0.1, K_segments=96,
                 n_experts=4, gating_topk=4, gating_temperature=1.0, gating_noise_std=0.0,
                 bias_update_speed=0.1, season_harmonics=2, season_resid_l2=1e-5,
                 cdf_mix_power=4.0, gate_dropout=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        # 保存配置
        self._y_mean_cfg = np.asarray(y_mean, dtype=np.float32).tolist()
        self._y_std_cfg  = np.asarray(y_std, dtype=np.float32).tolist()
        self._ctx_min_cfg = float(ctx_min)
        self._ctx_max_cfg = float(ctx_max)
        self._hidden_cfg = int(hidden)
        self._dropout_cfg = float(dropout)
        self._K_segments_cfg = int(K_segments)
        self._hour_bins_cfg = HOUR_BINS
        self._n_experts_cfg = int(n_experts)
        self._gating_topk_cfg = int(gating_topk)
        self._gating_temperature_cfg = float(gating_temperature)
        self._gating_noise_std_cfg = float(gating_noise_std)
        self._bias_update_speed_cfg = float(bias_update_speed)
        self._season_harmonics_cfg = int(season_harmonics)
        self._season_resid_l2_cfg = float(season_resid_l2)
        self._cdf_mix_power_cfg = float(cdf_mix_power)
        self.gate_dropout = float(gate_dropout)
        self.gate_dropout_layer = layers.Dropout(float(gate_dropout), name="gate_dropout")

        # 标准化
        self.norm_y = Log1pStandardize(y_mean, y_std, name="norm_y")
        self.norm_ctx = MinMaxNormalizeClip(ctx_min, ctx_max, name="norm_ctx")

        # 上下文嵌入
        self.e_event = layers.Embedding(NUM_EVENT_TYPES, 8, name="emb_event")
        self.e_band  = layers.Embedding(NUM_BANDS, 8, name="emb_band")
        self.e_tier  = layers.Embedding(NUM_TIERS, 8, name="emb_tier")
        self.e_server= layers.Embedding(NUM_SERVERS, 4, name="emb_server")
        self.ctx_dense = layers.Dense(64, activation="relu")

        # 历史编码器 (GRU)
        self.hist_gru = layers.GRU(hidden, return_sequences=False, name="hist_gru")
        self.hist_proj = layers.Dense(hidden, activation="relu")

        # 直接预测未来总和的头（共享 gate_in）
        self.sum_head = keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dense(1),                      # 输出对数域的未来总和
        ], name="sum_head")
        # MoE 形状头
        self.shape_moe = ShapeMoE(
            K=K_segments, hidden=hidden, n_experts=n_experts,
            gating_topk=gating_topk,
            gating_temperature=gating_temperature,
            gating_noise_std=gating_noise_std,
            bias_update_speed=bias_update_speed,
            season_harmonics=season_harmonics,
            season_resid_l2=season_resid_l2, season_resid_dropout=0.0,
            name="shape_moe"
        )

        # 季节网络
        self.season_hour = SeasonHourSegNet(K_segments=K_segments, hour_bins=HOUR_BINS, name="season_hour")
        self.weekhol_corr = WeekHolCorrectionNet(week_bins=WEEK_BINS, hol_bins=HOL_BINS, name="weekhol_corr")

        # 指标
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.time_loss_tracker = keras.metrics.Mean(name="time_loss")
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
            "n_experts": self._n_experts_cfg,
            "gating_temperature": self._gating_temperature_cfg,
            "gating_noise_std": self._gating_noise_std_cfg,
            "bias_update_speed": self._bias_update_speed_cfg,
            "season_harmonics": self._season_harmonics_cfg,
            "season_resid_l2": self._season_resid_l2_cfg,
            "gating_topk": self._gating_topk_cfg,
            "cdf_mix_power": self._cdf_mix_power_cfg,
            "gate_dropout": self.gate_dropout,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["y_mean"] = np.asarray(config["y_mean"], dtype=np.float32)
        config["y_std"] = np.asarray(config["y_std"], dtype=np.float32)
        return cls(**config)

    def call(self, inputs, training=False):
        # 与 CDFShapeMLP 类似，但使用 self.shape_moe 输出 u_mix 和 season_coef_mix
        x_hist = tf.cast(inputs["x_hist"], tf.float32)
        hist_len = tf.cast(inputs["hist_len"], tf.int32)
        future_len = tf.cast(inputs["future_len"], tf.int32)
        hh = tf.cast(inputs["exo_hist_hour"], tf.int32)
        hw = tf.cast(inputs["exo_hist_week"], tf.int32)
        hhol = tf.cast(inputs["exo_hist_hol"], tf.int32)
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

        # ------------ 进度分段（修正版：完全独立于 padding 长度） ------------
        th = tf.cast(hist_len, tf.float32)
        tf_len = tf.cast(future_len, tf.float32)
        T = tf.maximum(th + tf_len, 1.0)

        # 历史进度：从 1 到 Lh，归一化到 [0,1)，分母是每个样本自己的 th
        pos_h = tf.cast(tf.range(1, Lh + 1), tf.float32)[tf.newaxis, :]  # (1, Lh)
        p_hist = tf.clip_by_value(pos_h / T[:, tf.newaxis], 0.0, 1.0)   # (B, Lh)

        # 未来进度：未来步的绝对位置 = th + 1..th+Lf，除以 T
        pos_f = tf.cast(tf.range(1, Lf + 1), tf.float32)[tf.newaxis, :]  # (1, Lf)
        p_future = tf.clip_by_value((th[:, tf.newaxis] + pos_f) / T[:, tf.newaxis], 0.0, 1.0)  # (B, Lf)

        K = float(self._K_segments_cfg)
        seg_hist = tf.cast(tf.floor(p_hist * K), tf.int32)
        seg_hist = tf.clip_by_value(seg_hist, 0, self._K_segments_cfg - 1)
        seg_future = tf.cast(tf.floor(p_future * K), tf.int32)
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

        # 历史编码 (GRU)
        hist_feat = tf.concat([x_hist_norm, p_hist[..., tf.newaxis]], axis=-1)
        h_vec = self.hist_gru(hist_feat, mask=hist_mask, training=training)
        h_vec = self.hist_proj(h_vec)

        # 上下文（已移除 context_num）
        ctx_cat = tf.cast(inputs["context_cat"], tf.int32)
        e0 = self.e_event(ctx_cat[:, 0])
        e1 = self.e_band(ctx_cat[:, 1])
        e2 = self.e_tier(ctx_cat[:, 2])
        e3 = self.e_server(ctx_cat[:, 3])
        ctx_vec = tf.concat([e0, e1, e2, e3], axis=-1)   # 不再拼接 ctx_num
        ctx_vec = self.ctx_dense(ctx_vec)

        # MoE 门控输入（添加 dropout）
        gate_in = tf.concat([h_vec, ctx_vec], axis=-1)
        gate_in = self.gate_dropout_layer(gate_in, training=training)
        pi, u_exp, season_coef = self.shape_moe(gate_in, training=training)

        # 直接预测未来总和（对数域）
        log_future_sum = self.sum_head(gate_in)                  # (B, 1)
        direct_future_sum = tf.squeeze(tf.exp(log_future_sum), axis=-1) + 1e-6  # (B,)

        # 混合形状分布
        u_mix = tf.reduce_sum(pi[..., tf.newaxis] * u_exp, axis=1)   # (B, K)
        u_mix = u_mix / (tf.reduce_sum(u_mix, axis=-1, keepdims=True) + 1e-8)

        # --- CDF 推断总量（基础域） ---
        x_hist_ds_clipped = tf.clip_by_value(x_hist_ds, -5.0, 15.0)
        x_hist_base = tf.math.expm1(x_hist_ds_clipped) * hist_mask_f
        prefix_sum_base = tf.reduce_sum(x_hist_base, axis=[1,2])

        p0 = th / T
        upos = p0 * K
        j = tf.cast(tf.floor(upos), tf.int32)
        j = tf.clip_by_value(j, 0, self._K_segments_cfg - 1)
        frac = tf.clip_by_value(upos - tf.cast(j, tf.float32), 0.0, 1.0)

        cumsum_u = tf.cumsum(u_mix, axis=-1)
        prev = tf.gather(cumsum_u, tf.maximum(j - 1, 0), batch_dims=1)
        prev = tf.where(j > 0, prev, tf.zeros_like(prev))
        uj = tf.gather(u_mix, j, batch_dims=1)
        cdf_p0 = prev + frac * uj
        cdf_p0 = tf.clip_by_value(cdf_p0, 1e-4, 1.0 - 1e-6)

        total_hat_base_cdf = prefix_sum_base / cdf_p0
        total_hat_base_cdf = tf.clip_by_value(total_hat_base_cdf, 1e-4, 1e12)

        # 混合因子
        alpha = tf.clip_by_value(cdf_p0 ** self._cdf_mix_power_cfg, 0.0, 1.0)   # (B,)
        total_hat_direct = prefix_sum_base + direct_future_sum
        total_hat_base = alpha * total_hat_direct + (1.0 - alpha) * total_hat_base_cdf
        total_hat_base = tf.clip_by_value(total_hat_base, 1e-4, 1e12)

        # 分配未来量（基于 u_mix）
        one_hot_j = tf.one_hot(j, depth=self._K_segments_cfg, dtype=tf.float32)
        after_j = tf.cast(tf.range(self._K_segments_cfg)[tf.newaxis, :] > j[:, tf.newaxis], tf.float32)
        tail_frac_in_j = (1.0 - frac)[:, tf.newaxis]
        tail_mass = u_mix * after_j + u_mix * one_hot_j * tail_frac_in_j

        seg_onehot = tf.one_hot(seg_future, depth=self._K_segments_cfg, dtype=tf.float32)
        seg_onehot = seg_onehot * tf.cast(future_mask[..., tf.newaxis], tf.float32)
        n_per_seg = tf.reduce_sum(seg_onehot, axis=1) + 1e-6
        per_step_mass = tf.reduce_sum(seg_onehot * (tail_mass / n_per_seg)[:, tf.newaxis, :], axis=-1)
        per_step_mass = per_step_mass * tf.cast(future_mask, tf.float32)

        y_ds_base = total_hat_base[:, tf.newaxis] * per_step_mass
        y_ds_base = tf.maximum(y_ds_base, 0.0)

        # --- 专家季节性残差（傅里叶基） ---
        M = self._season_harmonics_cfg
        hour_bins = self._hour_bins_cfg
        fh_clean = tf.maximum(fh - 1, 0)                               # 转回 0‑based
        hour_f = tf.cast(fh_clean, tf.float32) / tf.cast(hour_bins, tf.float32)
        theta = 2.0 * np.pi * hour_f
        harmonics_range = tf.cast(tf.range(1, M + 1), tf.float32)
        theta_m = theta[..., tf.newaxis] * harmonics_range[tf.newaxis, tf.newaxis, :]
        basis = tf.reshape(
            tf.stack([tf.sin(theta_m), tf.cos(theta_m)], axis=-1),   # (B, Lf, M, 2)
            [B, Lf, 2 * M]
        )

        season_coef_bkne = tf.transpose(season_coef, [0, 2, 1, 3])  # (B, K, N, 2M)
        coef_step = tf.gather(season_coef_bkne, seg_future, axis=1, batch_dims=1)  # (B, Lf, N, 2M)
        residual_per_expert = tf.reduce_sum(coef_step * basis[:, :, tf.newaxis, :], axis=-1)
        season_resid_mix = tf.reduce_sum(residual_per_expert * pi[:, tf.newaxis, :], axis=-1)[..., tf.newaxis]
        season_resid_mix = season_resid_mix * tf.cast(future_mask3, tf.float32)
        season_resid_mix = tf.clip_by_value(season_resid_mix, -4.0, 4.0)

        s_future_eff = s_fut + season_resid_mix
        s_future_eff = tf.clip_by_value(s_future_eff, -8.0, 8.0)
        corr_fut = tf.clip_by_value(corr_fut, -3.0, 3.0)

        # --- 最终预测与星期/假日修正 ---
        log_y_ds = tf.math.log1p(y_ds_base)[..., tf.newaxis]
        mask_f = tf.cast(future_mask3, tf.float32)

        # 仅含小时效应的预测（用于形状损失）
        y_hat_no_corr = y_ds_base[..., tf.newaxis] * tf.exp(s_future_eff)
        y_hat_no_corr = tf.maximum(y_hat_no_corr, 0.0) * mask_f

        # corr_total / corr_step 分解
        corr_total = tf.reduce_sum(corr_fut * mask_f, axis=[1,2], keepdims=True) / (
            tf.reduce_sum(mask_f, axis=[1,2], keepdims=True) + 1e-6)
        corr_total = tf.clip_by_value(corr_total, -1.0, 1.0)
        corr_step = corr_fut - corr_total
        corr_step = tf.clip_by_value(corr_step, -2.0, 2.0)

        log_y_hat_base = log_y_ds + s_future_eff + corr_step
        log_y_hat_base = tf.clip_by_value(log_y_hat_base, -5.0, 15.0)

        y_hat_raw = tf.math.expm1(log_y_hat_base) * mask_f

        future_sum_raw = tf.reduce_sum(y_hat_raw, axis=[1,2])       # (B,)
        target_sum = future_sum_raw * tf.exp(tf.squeeze(corr_total, axis=[1,2]))
        scale = target_sum / (future_sum_raw + 1e-6)
        y_hat = y_hat_raw * scale[:, tf.newaxis, tf.newaxis]
        pred_log = safe_log1p(y_hat)

        return {
            "y_hat": y_hat,
            "y_hat_no_corr": y_hat_no_corr,
            "future_mask": future_mask3,
            "pred_log": pred_log,
            "s_future": s_fut,
            "corr_future": corr_fut,
            "pi": pi,
            "u_mix": u_mix,
            "cdf_p0": cdf_p0,
            "total_hat_base": total_hat_base,
            "prefix_sum_base": prefix_sum_base,
            "y_ds_base": y_ds_base,
        }
    def compute_loss(self, y_true, out, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        future_mask = out["future_mask"]
        mask_f = tf.cast(future_mask, tf.float32)

        log_true = safe_log1p(y_true)
        log_pred = out["pred_log"]

        # 点级损失 (每样本均值)
        per_elem = huber_per_elem(log_true, log_pred, delta=1.0)
        loss_time_i = tf.reduce_sum(per_elem * mask_f, axis=[1,2]) / (
            tf.reduce_sum(mask_f, axis=[1,2]) + 1e-8)      # (B,)

        # 总量损失 (每样本)
        sum_true = tf.reduce_sum(y_true * mask_f, axis=[1,2])
        sum_pred = tf.reduce_sum(out["y_hat"] * mask_f, axis=[1,2])
        loss_sum_i = huber_per_elem(safe_log1p(sum_true), safe_log1p(sum_pred),
                                    delta=1.0)             # (B,)

        # 形状 CDF 损失 (每样本)
        y_true_ds = tf.maximum(y_true - (tf.math.expm1(out["s_future"] + out["corr_future"])), 0.0) * mask_f
        y_pred_ds = out["y_ds_base"][..., tf.newaxis] * mask_f

        c_true = tf.cumsum(y_true_ds, axis=1)
        c_pred = tf.cumsum(y_pred_ds, axis=1)
        tot_true = tf.reduce_sum(y_true_ds, axis=1, keepdims=True) + 1e-6
        tot_pred = tf.reduce_sum(y_pred_ds, axis=1, keepdims=True) + 1e-6
        l1 = tf.abs(c_true / tot_true - c_pred / tot_pred)
        loss_shape_i = tf.reduce_sum(l1 * mask_f, axis=[1,2]) / (
            tf.reduce_sum(mask_f, axis=[1,2]) + 1e-8)      # (B,)

        # 组合总损失（不加权，后续在 train_step 中加权）
        loss_per_sample = 0.02 * loss_time_i + loss_sum_i + loss_shape_i

        if sample_weight is not None:
            sample_weight = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)
            loss_time = tf.reduce_sum(loss_time_i * sample_weight) / (tf.reduce_sum(sample_weight) + 1e-8)
            loss_sum = tf.reduce_sum(loss_sum_i * sample_weight) / (tf.reduce_sum(sample_weight) + 1e-8)
            loss_shape = tf.reduce_sum(loss_shape_i * sample_weight) / (tf.reduce_sum(sample_weight) + 1e-8)
            total_loss = tf.reduce_sum(loss_per_sample * sample_weight) / (tf.reduce_sum(sample_weight) + 1e-8)
        else:
            loss_time = tf.reduce_mean(loss_time_i)
            loss_sum = tf.reduce_mean(loss_sum_i)
            loss_shape = tf.reduce_mean(loss_shape_i)
            total_loss = tf.reduce_mean(loss_per_sample)

        return {
            "total_loss": total_loss,
            "time_loss": loss_time,
            "sum_loss": loss_sum,
            "shape_loss": loss_shape
        }
    
    def train_step(self, data):
        x, y_true, sw = data
        with tf.GradientTape() as tape:
            out = self(x, training=True)
            losses = self.compute_loss(y_true, out, sample_weight=sw)
        grads = tape.gradient(losses["total_loss"], self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # 更新专家偏置
        self.shape_moe.update_bias(out["pi"])

        # 更新所有指标
        self.loss_tracker.update_state(losses["total_loss"])
        self.time_loss_tracker.update_state(losses["time_loss"])
        self.sum_loss_tracker.update_state(losses["sum_loss"])
        self.shape_loss_tracker.update_state(losses["shape_loss"])

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y_true, sw = data
        out = self(x, training=False)
        losses = self.compute_loss(y_true, out, sample_weight=sw)

        # 更新所有指标
        self.loss_tracker.update_state(losses["total_loss"])
        self.time_loss_tracker.update_state(losses["time_loss"])
        self.sum_loss_tracker.update_state(losses["sum_loss"])
        self.shape_loss_tracker.update_state(losses["shape_loss"])

        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.loss_tracker, self.time_loss_tracker, self.sum_loss_tracker, self.shape_loss_tracker]