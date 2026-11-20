# evaluation/visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from data.constants import inv_eventType, tierListOfServer, serverNames, bandNames
# 直接复用训练用的序列加载函数
from data.sequences import load_sequences_from_hdf5
from data.slicing import _build_inputs_for_slices_dict


def slice_points(length, t_min, max_slices):
    points = np.arange(t_min, length, dtype=int)
    if max_slices is not None and len(points) > max_slices:
        points = points[np.linspace(0, len(points)-1, max_slices).round().astype(int)]
    return points.tolist()


def reconstruction_points(length, t_min):
    return sorted({max(t_min, min(length-1, round(p * length))) for p in (.2, .5, .8)})


def plot_sequence_slice_sums_batched_v2(
    model,
    seq: Dict[str, Any],
    *,
    t_min: int = 48,
    max_slices: int = 512,
    chunk_size: int = 256,
    out_path: str = None,
    title_prefix: str = "seq"
):
    """
    对单个序列在不同切点 t0 下的累计和预测进行可视化。
    """
    import tensorflow as tf
    y = seq["y"].astype(np.float32)
    T = y.shape[0]
    if T <= t_min + 1:
        return

    # 生成 t0 列表
    t0_list = slice_points(T, t_min, max_slices)

    # 真实累计和
    per_t_sum = y.sum(axis=1)
    prefix = np.concatenate([[0.0], np.cumsum(per_t_sum, dtype=np.float64)])
    full_true_sum = float(prefix[T])

    # 选取几个代表性切点用于显示重构段
    desired_t0s = reconstruction_points(T, t_min)

    segments = {}
    hist_sum_curve = []
    hist_plus_pred_curve = []
    pi_curve = []

    # 分块推理
    for s in range(0, len(t0_list), chunk_size):
        chunk = t0_list[s:s + chunk_size]
        x = _build_inputs_for_slices_dict(seq, chunk)
        x_tf = {k: tf.convert_to_tensor(v) for k, v in x.items()}
        out = model(x_tf, training=False)

        if "pi" in out:
            pi = out["pi"].numpy()
            pi_curve.extend(pi.tolist())

        y_hat = out["y_hat"].numpy()
        max_k = y_hat.shape[1]
        k_list = x["future_len"]
        mask = (np.arange(max_k)[None, :] < k_list[:, None]).astype(np.float32)
        y_hat = y_hat * mask[:, :, None]

        pred_future_sum = y_hat.sum(axis=(1, 2)).astype(np.float64)
        step_sum = y_hat.sum(axis=2).astype(np.float64)

        for b, t0 in enumerate(x["hist_len"].tolist()):
            if t0 in desired_t0s and t0 not in segments:
                increments = step_sum[b, :int(k_list[b])]
                segments[t0] = (t0, prefix[t0] + np.r_[0., np.cumsum(increments, dtype=np.float64)])

        hist_sum = prefix[x["hist_len"].astype(np.int32)].astype(np.float64)
        hist_sum_curve.extend(hist_sum.tolist())
        hist_plus_pred_curve.extend((hist_sum + pred_future_sum).tolist())
    plot_slice_forecasts(prefix, t0_list, hist_sum_curve, hist_plus_pred_curve, full_true_sum,
                         segments, title_prefix, out_path, pi_curve=pi_curve)


def plot_slice_forecasts(prefix, t0_list, hist_sum_curve, hist_plus_pred_curve, full_true_sum,
                         segments, title_prefix, out_path, *, pi_curve=(), show_error=False):
    """Plot rolling endpoints and forecast segments, each beginning at its actual anchor."""
    T = len(prefix)-1
    has_pi = len(pi_curve) > 0
    if has_pi or show_error:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

    t_all = np.arange(0, T + 1, dtype=np.int32)
    ax1.plot(t_all, prefix, label="实际累计分数" if show_error else "true_prefix_sum")
    ax1.plot(t0_list, hist_sum_curve, label="起报时最新档线" if show_error else "hist_sum (true prefix at t0)")
    ax1.plot(t0_list, hist_plus_pred_curve, label="逐次预测的结算分数" if show_error else "hist_sum + pred_future_sum")
    ax1.plot(t0_list, np.full(len(t0_list), full_true_sum), label="真实结算分数" if show_error else "full_true_sum")

    for t0, (anchor, seg_prefix) in segments.items():
        t_seg = np.arange(anchor, anchor + len(seg_prefix), dtype=np.int32)
        pct = int(round(100 * t0 / max(1, T)))
        label = f"进度 {pct}% 时预测的走势" if show_error else f"pred-prefix seg @t0={t0} (~{pct}%)"
        ax1.plot(t_seg, seg_prefix, linestyle="--", label=label)

    ax1.set_ylabel("累计分数（分）" if show_error else "cumulative sum / total sum")
    if show_error:
        ax1.legend(ncol=2, fontsize=9)
    else:
        ax1.legend()
    ax1.set_title(title_prefix if show_error else f"{title_prefix}: slice-sum + recon prefix (T={T})")

    if has_pi:
        pi_arr = np.array(pi_curve, dtype=np.float64)
        n_experts = pi_arr.shape[1]
        for e in range(n_experts):
            ax2.plot(t0_list, pi_arr[:, e], label=f"pi[e{e}]")
        ax2.set_ylabel("routing pi")
        ax2.set_xlabel("slice point t0")
        ax2.set_ylim(0.0, 1.0)
        t0_to_idx = {int(t0): i for i, t0 in enumerate(t0_list)}
        for t0 in segments:
            i = t0_to_idx[t0]
            ex = int(np.argmax(pi_arr[i]))
            ax2.axvline(t0, linestyle=":", alpha=0.25)
            ax2.text(t0, 1.02, f"e{ex}", rotation=90, va="bottom", ha="center", fontsize=8)
        ax2.legend(ncol=min(n_experts, 4), fontsize=8, loc="upper right")
    elif show_error:
        error = np.asarray(hist_plus_pred_curve) - full_true_sum
        if full_true_sum > 0:
            error = 100. * error / full_true_sum
        ax2.axhline(0, lw=1, color='0.35')
        ax2.plot(t0_list, error, color='tab:green')
        ax2.set_ylabel('结算偏差（%）' if full_true_sum > 0 else '结算偏差（分）')
        ax2.set_xlabel('距开场的半小时数（偏差为正表示高估，为负表示低估）')
        ax2.grid(alpha=.2)
    else:
        ax1.set_xlabel("slice point t0")

    fig.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def visualize_val_sequences_after_training(
    model,
    raw_h5: str,
    out_dir: str,
    *,
    n_seq: int = 6,
    split: str = "val",
    t_min: int = 48,
    max_slices: int = 512,
    chunk_size: int = 256,
    seed: int = 1,
):
    """
    从 raw_h5 中随机选取 n_seq 条验证序列，生成切片可视化图。
    使用 data.sequences.load_sequences_from_hdf5 读取原始数据。
    """
    if not os.path.exists(raw_h5):
        print(f"[viz] raw_h5 not found: {raw_h5}. Skip visualization.")
        return

    seqs = load_sequences_from_hdf5(raw_h5, split=split)
    if not seqs:
        print(f"[viz] No sequences found in {raw_h5} for split={split}. Skip.")
        return

    rng = np.random.default_rng(seed)
    pick = list(range(len(seqs)))
    if len(pick) > n_seq:
        pick = rng.choice(pick, size=n_seq, replace=False).tolist()
        pick.sort()

    viz_dir = os.path.join(out_dir, "val_slice_viz")
    os.makedirs(viz_dir, exist_ok=True)

    for i, idx in enumerate(pick):
        seq = seqs[idx]
        ctx = seq["context_cat"].astype(int).tolist()
        eventType, band, tier, server, eventId = ctx[0], ctx[1], ctx[2], ctx[3], ctx[4]

        # 使用之前定义的映射
        event_str = inv_eventType.get(eventType, f"ev{eventType}")
        band_str = bandNames.get(band, f"b{band}")
        tier_val = seq.get("tier")
        if tier_val is None:
            tier_val = tierListOfServer[3][tier] if 0 <= tier < len(tierListOfServer[3]) else f"?{tier}"
        server_str = serverNames.get(server, "server?")

        title = f"val ctx(ev{eventId},{event_str},{band_str},t{tier_val},{server_str})"
        out_path = os.path.join(viz_dir,
                                f"{title}.png")
        try:
            plot_sequence_slice_sums_batched_v2(
                model, seq,
                t_min=t_min, max_slices=max_slices, chunk_size=chunk_size,
                out_path=out_path, title_prefix=title,
            )
            print(f"[viz] saved: {out_path}")
        except Exception as e:
            print(f"[viz] failed on seq idx={idx}: {e}")
