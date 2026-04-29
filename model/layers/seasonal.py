# model/layers/seasonal.py
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from utils.tf_ops import make_positions_tf   # 如果不需要可移除，但保留以备未来扩展

register_keras_serializable = keras.utils.register_keras_serializable


@register_keras_serializable(package="BangDream")
class SeasonHourSegNet(layers.Layer):
    """
    Hour seasonal effect with a global component + a segment-conditioned residual.

    Models:
        s(hour, week, holiday, seg) = s_global(hour) + s_residual(hour, week, holiday, seg)

    - s_global depends only on hour (shared across all segments).
    - s_residual is conditioned on segment via a small embedding and MLP,
      with extra regularization (dropout + L2) so it only learns deviations.
      A seg-dependent scale further modulates residual amplitude per segment.
    Output is a log-domain offset: (B, L, 1).
    """

    def __init__(
        self,
        K_segments: int,
        hour_bins: int = 48,
        week_bins: int = 7,
        hol_bins: int = 2,
        emb_hour: int = 8,
        emb_seg: int | None = None,
        emb_hour_resid: int | None = None,
        emb_week_resid: int = 4,
        emb_hol_resid: int = 2,
        hidden: int = 64,
        resid_hidden: int | None = None,
        resid_dropout: float = 0.10,
        resid_l2: float = 1e-5,
        resid_scale: float = 1.0,
        scale_max: float = 2.0,
        scale_hidden: int | None = None,
        scale_dropout: float = 0.05,
        scale_l2: float | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.K = int(K_segments)
        self.hour_bins = int(hour_bins)
        self.hour_stride = self.hour_bins + 1  # include padding 0

        self.week_bins = int(week_bins)
        self.hol_bins = int(hol_bins)
        self.week_stride = self.week_bins + 1  # include padding 0
        self.hol_stride = self.hol_bins + 1    # include padding 0

        emb_seg = int(emb_seg if emb_seg is not None else max(4, emb_hour // 2))
        emb_hour_resid = int(emb_hour_resid if emb_hour_resid is not None else max(4, emb_hour // 2))
        resid_hidden = int(resid_hidden if resid_hidden is not None else max(16, hidden // 2))
        scale_hidden = int(scale_hidden if scale_hidden is not None else max(8, emb_seg))
        scale_l2 = float(scale_l2 if scale_l2 is not None else resid_l2)
        self.scale_max = float(scale_max)

        # Global hour embedding (shared across segments)
        self.e_hour_global = layers.Embedding(self.hour_stride, emb_hour, name="season_emb_hour_global")

        # Residual path uses separate, smaller embeddings
        self.e_hour_resid = layers.Embedding(self.hour_stride, emb_hour_resid, name="season_emb_hour_resid")
        self.e_seg = layers.Embedding(self.K, emb_seg, name="season_emb_seg")
        self.e_week_resid = layers.Embedding(self.week_stride, emb_week_resid, name="season_emb_week_resid")
        self.e_hol_resid = layers.Embedding(self.hol_stride, emb_hol_resid, name="season_emb_hol_resid")

        # Global MLP (hour-only)
        self.mlp_global = keras.Sequential([
            layers.Dense(hidden, activation="relu"),
            layers.Dense(hidden // 2, activation="relu"),
            layers.Dense(1),
        ], name="season_hour_global_mlp")

        # Segment-dependent residual scale
        self.seg_scale_mlp = keras.Sequential([
            layers.Dense(
                scale_hidden,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(scale_l2),
            ),
            layers.Dropout(scale_dropout),
            layers.Dense(1, kernel_regularizer=keras.regularizers.l2(scale_l2)),
        ], name="season_seg_scale_mlp")

        # Residual MLP (hour + seg), regularized and scaled
        self.mlp_resid = keras.Sequential([
            layers.Dense(
                resid_hidden,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(resid_l2),
            ),
            layers.Dropout(resid_dropout),
            layers.Dense(
                resid_hidden // 2,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(resid_l2),
            ),
            layers.Dropout(resid_dropout),
            layers.Dense(1, kernel_regularizer=keras.regularizers.l2(resid_l2)),
        ], name="season_hour_seg_resid_mlp")

        self.resid_scale = float(resid_scale)

    def call(self, hour, week, hol, seg, mask):
        """
        hour: (B, L) int32 in [0..hour_bins], where 0 is padding
        week: (B, L) int32 in [0..week_bins], where 0 is padding
        hol : (B, L) int32 in [0..hol_bins], where 0 is padding
        seg : (B, L) int32 in [0..K-1]
        mask: (B, L) bool / 0-1
        """
        hour = tf.cast(hour, tf.int32)
        seg = tf.cast(seg, tf.int32)
        week = tf.cast(week, tf.int32)
        hol = tf.cast(hol, tf.int32)
        mask_f = tf.cast(mask, tf.float32)

        # Use hour-1 for cyclic features (0 is padding)
        hour_f = tf.cast(tf.maximum(hour - 1, 0), tf.float32) / tf.cast(self.hour_bins, tf.float32)
        sin_h = tf.sin(2.0 * np.pi * hour_f)
        cos_h = tf.cos(2.0 * np.pi * hour_f)

        # Clip week/holiday indices to valid ranges
        week = tf.clip_by_value(week, 0, self.week_bins)
        hol = tf.clip_by_value(hol, 0, self.hol_bins)

        # Global hour features
        eh_g = self.e_hour_global(hour)
        feats_g = tf.concat([eh_g, sin_h[..., tf.newaxis], cos_h[..., tf.newaxis]], axis=-1)
        out_g = self.mlp_global(feats_g)

        # Residual (seg-conditioned) features
        eh_r = self.e_hour_resid(hour)
        es = self.e_seg(seg)
        ew = self.e_week_resid(week)
        ehol = self.e_hol_resid(hol)
        feats_r = tf.concat([eh_r, es, ew, ehol, sin_h[..., tf.newaxis], cos_h[..., tf.newaxis]], axis=-1)
        out_r_base = self.mlp_resid(feats_r) * self.resid_scale  # (B,L,1)
        # seg-dependent scale in (0, scale_max)
        scale = self.scale_max * tf.sigmoid(self.seg_scale_mlp(es))  # (B,L,1)
        out_r = out_r_base * scale

        out = (out_g + out_r) * mask_f[..., tf.newaxis]
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "K_segments": self.K,
            "hour_bins": self.hour_bins,
            "week_bins": self.week_bins,
            "hol_bins": self.hol_bins,
            "emb_hour": self.e_hour_global.output_dim,
            "emb_seg": self.e_seg.output_dim,
            "emb_hour_resid": self.e_hour_resid.output_dim,
            "emb_week_resid": self.e_week_resid.output_dim,
            "emb_hol_resid": self.e_hol_resid.output_dim,
            "hidden": self.mlp_global.layers[0].units,
            "resid_hidden": self.mlp_resid.layers[0].units,
            "resid_dropout": self.mlp_resid.layers[1].rate,
            "resid_l2": self.mlp_resid.layers[0].kernel_regularizer.l2 if self.mlp_resid.layers[0].kernel_regularizer else 0.0,
            "resid_scale": self.resid_scale,
            "scale_max": self.scale_max,
            "scale_hidden": self.seg_scale_mlp.layers[0].units,
            "scale_dropout": self.seg_scale_mlp.layers[1].rate,
            "scale_l2": self.seg_scale_mlp.layers[0].kernel_regularizer.l2 if self.seg_scale_mlp.layers[0].kernel_regularizer else 0.0,
        })
        return config


@register_keras_serializable(package="BangDream")
class WeekHolCorrectionNet(layers.Layer):
    """
    Weekday/holiday correction applied at the end (after base shape+hour effects),
    then renormalized to keep the inferred future sum unchanged.
    Output is a log-factor (B,L,1).
    """

    def __init__(self, week_bins=7, hol_bins=2, emb_week=4, emb_hol=2, hidden=32, **kwargs):
        super().__init__(**kwargs)
        self.week_bins = int(week_bins)
        self.hol_bins = int(hol_bins)

        self.e_week = layers.Embedding(self.week_bins + 1, emb_week, name="corr_emb_week")
        self.e_hol = layers.Embedding(self.hol_bins + 1, emb_hol, name="corr_emb_hol")

        self.mlp = keras.Sequential([
            layers.Dense(hidden, activation="relu"),
            layers.Dense(1, activation=None),
        ], name="weekhol_corr_mlp")

    def call(self, week, hol, mask):
        week = tf.cast(week, tf.int32)
        hol = tf.cast(hol, tf.int32)

        idx_week = tf.clip_by_value(week, 0, self.week_bins)
        idx_hol = tf.clip_by_value(hol, 0, self.hol_bins)

        ew = self.e_week(idx_week)
        eh = self.e_hol(idx_hol)
        feats = tf.concat([ew, eh], axis=-1)

        out = self.mlp(feats)  # (B,L,1)
        out = out * tf.cast(mask[..., tf.newaxis], tf.float32)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "week_bins": self.week_bins,
            "hol_bins": self.hol_bins,
            "emb_week": self.e_week.output_dim,
            "emb_hol": self.e_hol.output_dim,
            "hidden": self.mlp.layers[0].units,
        })
        return config