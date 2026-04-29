# compare_predictions.py
import os
import argparse
import numpy as np
import tensorflow as tf


from data.slicing import _build_inputs_for_slices_dict
from data.raw import generate_sequences
from service.predictor import Predictor

def load_model(model_path: str):
    from model.moe_shape import MoeShapeModel
    from model.layers.moe import ShapeMoE
    from model.layers.seasonal import SeasonHourSegNet, WeekHolCorrectionNet
    from model.layers.normalization import Log1pStandardize, MinMaxNormalizeClip
    return tf.keras.models.load_model(model_path, custom_objects={
        'MoeShapeModel': MoeShapeModel,
        'ShapeMoE': ShapeMoE,
        'SeasonHourSegNet': SeasonHourSegNet,
        'WeekHolCorrectionNet': WeekHolCorrectionNet,
        'Log1pStandardize': Log1pStandardize,
        'MinMaxNormalizeClip': MinMaxNormalizeClip,
    })

def compute_curve_from_model(model, seq: dict, t0: int) -> dict:
    """使用与可视化完全相同的方式计算曲线"""
    x = _build_inputs_for_slices_dict(seq, [t0])
    x_tf = {k: tf.convert_to_tensor(v) for k, v in x.items()}
    out = model(x_tf, training=False)
    y_hat = out["y_hat"].numpy()[0, :, 0]
    future_len = int(x["future_len"][0])
    y_hist = seq["y"][:t0].flatten().astype(np.float64)
    hist_sum = y_hist.sum()
    future_sum = y_hat[:future_len].sum()
    total_sum_hat = hist_sum + future_sum
    cum_hist = np.cumsum(y_hist)
    cum_future = np.cumsum(y_hat[:future_len]) + (cum_hist[-1] if t0 > 0 else 0.0)
    prefix_curve = np.concatenate([cum_hist, cum_future]).tolist()
    return {
        "t0": t0,
        "future_len": future_len,
        "y_hat": y_hat[:future_len].tolist(),
        "total_sum_hat": float(total_sum_hat),
        "future_sum_hat": float(future_sum),
        "prefix_curve": prefix_curve,
    }

def main():
    parser = argparse.ArgumentParser(description="对比服务端与模型直接推理的预测")
    parser.add_argument("--model_path", type=str, default="saved_models/moe.keras")
    parser.add_argument("--raw_dir", type=str, default="datasets/raw")
    parser.add_argument("--event_id", type=int, default=306)
    parser.add_argument("--server", type=int, default=3)
    parser.add_argument("--tier", type=int, default=1000)
    parser.add_argument("--t_min", type=int, default=48)
    args = parser.parse_args()

    print("加载模型...")
    model = load_model(args.model_path)
    predictor = Predictor(args.model_path)

    print("生成序列...")
    seqs = generate_sequences(
        event_ids=[args.event_id],
        server_list=[args.server],
        tier_list=[args.tier],
        raw_dir=args.raw_dir,
        split_threshold=10000,
    )
    if not seqs:
        print("未找到序列。")
        return
    seq = seqs[0]
    T = len(seq["y"])
    if T <= args.t_min:
        print("序列太短。")
        return

    t0_list = list(range(args.t_min, T))
    max_diff_sum = 0.0
    max_diff_yhat = 0.0
    for t0 in t0_list:
        srv = predictor.predict_current_curve(seq, t0)
        viz = compute_curve_from_model(model, seq, t0)
        diff_sum = abs(srv["future_sum_hat"] - viz["future_sum_hat"])
        diff_y = np.max(np.abs(np.array(srv["y_hat"]) - np.array(viz["y_hat"]))) if srv["y_hat"] else 0.0
        if diff_sum > max_diff_sum:
            max_diff_sum = diff_sum
        if diff_y > max_diff_yhat:
            max_diff_yhat = diff_y

    print(f"对比完成，共 {len(t0_list)} 个切点。")
    print(f"最大 future_sum_hat 差异: {max_diff_sum:.6f}")
    print(f"最大 y_hat 差异: {max_diff_yhat:.6f}")
    if max_diff_sum > 1e-4 or max_diff_yhat > 1e-4:
        print("存在显著差异，请检查 Predictor 与模型加载时的配置是否一致。")
    else:
        print("预测完全一致。")

if __name__ == "__main__":
    main()