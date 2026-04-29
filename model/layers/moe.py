# model/layers/moe.py
import tensorflow as tf
import keras
from keras import layers
@keras.saving.register_keras_serializable(package="BangDream")
class ShapeMoE(layers.Layer):
    """Mixture-of-Experts with dynamic expert bias (loss-free balancing)."""
    def __init__(
        self,
        K: int = 96,
        hidden: int = 128,
        n_experts: int = 4,
        gating_temperature: float = 1.0,
        gating_topk: int = 0,
        gating_noise_std: float = 0.0,
        bias_update_speed: float = 0.1,
        bias_init: float = 0.0,
        season_harmonics: int = 2,
        season_resid_l2: float = 0.0,
        season_resid_dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.K = int(K)
        self.n_experts = int(n_experts)
        self.season_harmonics = int(season_harmonics)
        self.season_coef_dim = 2 * self.season_harmonics
        self.gating_temperature = float(gating_temperature)
        self.gating_topk = int(gating_topk)
        self.gating_noise_std = float(gating_noise_std)
        self.bias_update_speed = float(bias_update_speed)
        self.season_resid_l2 = float(season_resid_l2)
        self.season_resid_dropout = float(season_resid_dropout)

        # 门控网络
        self.gate = keras.Sequential([
            layers.Dense(hidden, activation="relu"),
            layers.Dense(hidden, activation="relu"),
            layers.Dense(self.n_experts, activation=None),
        ], name="gating_net")

        # 专家偏置（手动更新，不可训练）
        self.expert_bias = self.add_weight(
            name="expert_bias",
            shape=(self.n_experts,),
            initializer=keras.initializers.Constant(float(bias_init)),
            trainable=False,
            dtype=tf.float32
        )

        # 专家主干和头
        self.expert_trunks = [
            keras.Sequential([
                layers.Dense(hidden, activation="relu"),
                layers.Dense(hidden, activation="relu"),
            ], name=f"shape_expert{i}_trunk")
            for i in range(self.n_experts)
        ]
        self.expert_u_heads = [
            layers.Dense(self.K, activation=None, name=f"shape_expert{i}_u_head")
            for i in range(self.n_experts)
        ]
        # 季节系数头（可选）
        coef_reg = keras.regularizers.l2(self.season_resid_l2) if self.season_resid_l2 > 0 else None
        self.expert_season_heads = [
            layers.Dense(self.K * self.season_coef_dim, activation=None,
                         kernel_regularizer=coef_reg,
                         name=f"shape_expert{i}_season_head")
            for i in range(self.n_experts)
        ]

    def call(self, h_and_ctx, training=None):
        B = tf.shape(h_and_ctx)[0]
        logits = self.gate(h_and_ctx)                         # (B, N)

        # 加上动态偏置
        biased_logits = logits + self.expert_bias[tf.newaxis, :]

        # 噪声注入（仅训练时）
        if training and self.gating_noise_std > 0:
            biased_logits += tf.random.normal(tf.shape(biased_logits),
                                              stddev=self.gating_noise_std)

        # 温度缩放
        biased_logits = biased_logits / max(1e-6, float(self.gating_temperature))

        # 可选 top‑k（暂未启用）
        if self.gating_topk > 0 and self.gating_topk < self.n_experts:
            values, indices = tf.math.top_k(biased_logits, k=self.gating_topk, sorted=False)
            mask = tf.reduce_sum(tf.one_hot(indices, depth=self.n_experts, dtype=biased_logits.dtype), axis=-2)
            biased_logits = tf.where(mask > 0, biased_logits, tf.cast(-1e9, biased_logits.dtype))

        pi = tf.nn.softmax(biased_logits, axis=-1)            # (B, N)

        # 专家输出
        u_list, coef_list = [], []
        for i in range(self.n_experts):
            h_e = self.expert_trunks[i](h_and_ctx, training=training)
            u_e = tf.nn.softmax(self.expert_u_heads[i](h_e), axis=-1)
            # 可选 dropout（仅训练时）
            if training and self.season_resid_dropout > 0:
                h_e = tf.nn.dropout(h_e, rate=self.season_resid_dropout)
            coef_e_flat = self.expert_season_heads[i](h_e)
            coef_e = tf.reshape(coef_e_flat, [B, self.K, self.season_coef_dim])
            u_list.append(u_e)
            coef_list.append(coef_e)

        u = tf.stack(u_list, axis=1)                         # (B, N, K)
        season_coef = tf.stack(coef_list, axis=1)            # (B, N, K, 2M)
        return pi, u, season_coef

    def update_bias(self, pi):
        """根据当前 batch 的平均路由概率更新专家偏置（EMA）。"""
        avg_pi = tf.reduce_mean(pi, axis=0)                  # (N,)
        target = tf.ones_like(avg_pi) / float(self.n_experts)
        error = avg_pi - target
        new_bias = self.expert_bias - self.bias_update_speed * error
        self.expert_bias.assign(new_bias)

    def get_config(self):
        config = super().get_config()
        config.update({
            "K": self.K,
            "n_experts": self.n_experts,
            "season_harmonics": self.season_harmonics,
            "gating_temperature": self.gating_temperature,
            "gating_topk": self.gating_topk,
            "gating_noise_std": self.gating_noise_std,
            "bias_update_speed": self.bias_update_speed,
            "bias_init": float(self.expert_bias.numpy()[0]),  # 假设所有偏置初始值相同
            "season_resid_l2": self.season_resid_l2,
            "season_resid_dropout": self.season_resid_dropout,
        })
        return config