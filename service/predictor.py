# service/predictor.py
import os
import numpy as np
import tensorflow as tf

# 强制使用 CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# tf.config.set_visible_devices([], 'GPU')

from data.slicing import _build_inputs_for_slices_dict

class Predictor:
    def __init__(self, model_path: str):
        # 导入自定义层，确保注册
        from model.moe_shape import MoeShapeModel
        from model.layers.moe import ShapeMoE
        from model.layers.seasonal import SeasonHourSegNet, WeekHolCorrectionNet
        from model.layers.normalization import Log1pStandardize, MinMaxNormalizeClip

        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'MoeShapeModel': MoeShapeModel,
                'ShapeMoE': ShapeMoE,
                'SeasonHourSegNet': SeasonHourSegNet,
                'WeekHolCorrectionNet': WeekHolCorrectionNet,
                'Log1pStandardize': Log1pStandardize,
                'MinMaxNormalizeClip': MinMaxNormalizeClip,
            }
        )

    def predict_slices(self, seq: dict, t0_list: list, batch_size: int = 256) -> list:
        """
        批量推理：返回每个切点的未来总和与全序列原始域总量。
        不再依赖 total_hat_base，只使用 y_hat 求和。
        """
        results = []
        y_full = seq["y"].flatten().astype(np.float64)   # 整条序列的原始值
        T = len(y_full)

        for start in range(0, len(t0_list), batch_size):
            chunk = t0_list[start:start + batch_size]
            x = _build_inputs_for_slices_dict(seq, chunk)
            x_tf = {k: tf.convert_to_tensor(v) for k, v in x.items()}
            out = self.model(x_tf, training=False)

            y_hat_batch = out["y_hat"].numpy()            # (B, Lf, 1)
            future_lens = x["future_len"]                 # (B,)

            for i, t0 in enumerate(chunk):
                t0 = int(t0)
                k = int(future_lens[i])
                future_sum = float(y_hat_batch[i, :k, 0].sum())
                hist_sum = float(y_full[:t0].sum())
                total_sum = hist_sum + future_sum

                results.append({
                    "t0": t0,
                    "future_sum_hat": future_sum,
                    "total_sum_hat": total_sum,
                })
        return results

    def predict_current_curve(self, seq: dict, t0: int) -> dict:
        """
        获取指定切点下的完整预测，并返回：
        - y_hat: 未来步预测值（原始单位）
        - future_len: 未来步数
        - total_sum_hat: 全序列预测总消费（已知历史 + 预测未来）
        - prefix_curve: 从第1步到序列结束的累积和曲线（真实历史+预测未来）
        """
        x = _build_inputs_for_slices_dict(seq, [t0])
        x_tf = {k: tf.convert_to_tensor(v) for k, v in x.items()}
        out = self.model(x_tf, training=False)
        y_hat = out["y_hat"].numpy()[0, :, 0]       # (Lf,)
        future_len = int(x["future_len"][0])
        # 真实历史值 (原始单位)
        y_hist = seq["y"][:t0].flatten().astype(np.float64)   # (t0,)
        hist_sum = y_hist.sum()
        future_sum = y_hat[:future_len].sum()
        total_sum_hat = hist_sum + future_sum

        # 累积和曲线
        cum_hist = np.cumsum(y_hist)                           # 长度 t0
        cum_future = np.cumsum(y_hat[:future_len]) + (cum_hist[-1] if t0 > 0 else 0.0)
        prefix_curve = np.concatenate([cum_hist, cum_future]).tolist()  # 长度 T

        return {
            "t0": t0,
            "future_len": future_len,
            "y_hat": list(map(float, y_hat[:future_len].tolist())),
            "total_sum_hat": int(total_sum_hat),
            "future_sum_hat": int(future_sum),          # 方便单独使用
            "prefix_curve": prefix_curve,
        }