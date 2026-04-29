# data/preprocessing.py
import numpy as np
import h5py
from typing import List, Dict, Any
from .sequences import load_sequences_from_hdf5, compute_y_normalization_params, compute_contextnum_params
from .slicing import choose_t0_lists, build_query

def bucket_index(L: int, boundaries: List[int]) -> int:
    return int(np.searchsorted(np.asarray(boundaries, dtype=np.int32), L, side="right"))

def round_up(x: int, multiple: int = 32) -> int:
    return int(((x + multiple - 1) // multiple) * multiple)

def compute_sample_weight(K: int, T: int, mode: str = "sqrt_floor", floor: float = 0.2) -> np.float32:
    r = float(K) / float(T)
    if mode == "none":
        w = 1.0
    elif mode == "sqrt":
        w = np.sqrt(r)
    elif mode == "linear":
        w = r
    elif mode == "old":
        w = np.sqrt(r) * r
    elif mode == "sqrt_floor":
        w = max(np.sqrt(r), float(floor))
    elif mode == "linear_floor":
        w = max(r, float(floor))
    else:
        raise ValueError(f"Unknown weight mode: {mode}")
    return np.float32(w)

def compute_tier_weight(tier_idx: int,
                        center: int = 9,
                        alpha: float = 0.15,
                        floor: float = 0.3) -> float:
    """
    以 center 为中心，指数衰减的档位权重。
    alpha 越大，衰减越快；floor 为最低权重。
    """
    d = abs(int(tier_idx) - int(center))
    w = max(float(np.exp(-alpha * d)), float(floor))
    return w

def preprocess_split_to_groups(
    sequences: List[Dict[str, Any]],
    t0_lists: List[List[int]],
    out_group: h5py.Group,
    *,
    t_min: int,
    x_dim: int,
    context_cat_len: int,
    bucket_boundaries: List[int],
    length_mode: str = "hist",
    weight_mode: str = "sqrt_floor",
    weight_floor: float = 0.2,
    tier_center=9, tier_alpha=0.15, tier_floor=0.3,   # 新增档位加权参数
):
    assert length_mode == "hist", "仅支持按 hist 长度分桶"

    nbuckets = len(bucket_boundaries) + 1
    counts = np.zeros(nbuckets, dtype=np.int64)
    max_th_per_bucket = np.zeros(nbuckets, dtype=np.int32)
    max_tk_per_bucket = np.zeros(nbuckets, dtype=np.int32)
    global_max_th = 0

    # ---------- Pass 1: 统计 ----------
    for seq, t0_list in zip(sequences, t0_lists):
        T = len(seq["y"])
        for t0 in t0_list:
            th = int(t0)
            tk = int(T - t0)
            b = bucket_index(th, bucket_boundaries)
            counts[b] += 1
            if th > max_th_per_bucket[b]:
                max_th_per_bucket[b] = th
            if tk > max_tk_per_bucket[b]:
                max_tk_per_bucket[b] = tk
            if th > global_max_th:
                global_max_th = th

    # 确定各桶 Lh / Lf
    Lh_list = []
    Lf_list = []
    last_Lh = round_up(global_max_th, 32)
    for i in range(nbuckets):
        if i < len(bucket_boundaries):
            base_h = int(bucket_boundaries[i])
        else:
            base_h = int(last_Lh)
        Lh = max(base_h, round_up(max_th_per_bucket[i], 32))
        Lf = round_up(max_tk_per_bucket[i], 32) if counts[i] > 0 else 0
        Lh_list.append(int(Lh))
        Lf_list.append(int(Lf))

    # ---------- 分配数组 ----------
    buckets = []
    for b in range(nbuckets):
        n = int(counts[b])
        if n == 0:
            buckets.append(None)
            continue
        Lh = Lh_list[b]
        Lf = Lf_list[b] if Lf_list[b] > 0 else 32
        bk = {
            "n": n,
            "Lh": Lh,
            "Lf": Lf,
            "hist_len":   np.zeros((n,), np.int32),
            "future_len": np.zeros((n,), np.int32),
            "x_hist": np.zeros((n, Lh, x_dim), np.float32),
            "exo_hist_hour": np.zeros((n, Lh), np.int32),
            "exo_hist_week": np.zeros((n, Lh), np.int32),
            "exo_hist_hol":  np.zeros((n, Lh), np.int32),
            "context_cat": np.zeros((n, context_cat_len), np.int32),
            "context_num": np.zeros((n,), np.float32),
            "query_time": np.zeros((n, Lf, 2), np.float32),
            "future_exo_hour": np.zeros((n, Lf), np.int32),
            "future_exo_week": np.zeros((n, Lf), np.int32),
            "future_exo_hol":  np.zeros((n, Lf), np.int32),
            "y_future": np.zeros((n, Lf, x_dim), np.float32),
            "weight": np.zeros((n, 1), np.float32),
        }
        buckets.append(bk)

    # ---------- Pass 2: 填充 ----------
    write_ptr = np.zeros(nbuckets, dtype=np.int64)
    for seq, t0_list in zip(sequences, t0_lists):
        y = seq["y"].astype(np.float32)
        exo = seq["exo"].astype(np.int32)
        ctx_cat = (seq["context_cat"].astype(np.int32) + 1)  # 偏移，0 留给 padding
        ctx_num = np.float32(seq["context_num"])
        T = y.shape[0]

        for t0 in t0_list:
            th = int(t0)
            tk = int(T - t0)
            b = bucket_index(th, bucket_boundaries)
            bk = buckets[b]
            if bk is None:
                continue
            i = int(write_ptr[b])
            write_ptr[b] += 1

            Lh = bk["Lh"]
            Lf = bk["Lf"]
            if th > Lh or tk > Lf:
                raise RuntimeError(f"Bucket {b}: 长度超限 (th={th},tk={tk})")

            base_w = compute_sample_weight(tk, T, mode=weight_mode, floor=weight_floor)

            # 提取当前样本的 tier 索引（context_cat 已整体 +1，所以需减1恢复0-based）
            # ctx_cat 变量在该循环中已定义：ctx_cat = (seq["context_cat"].astype(np.int32) + 1)
            tier_idx = int(ctx_cat[2]) - 1   # 索引2为tier，0-based
            tier_w = compute_tier_weight(tier_idx, center=tier_center, alpha=tier_alpha, floor=tier_floor)

            bk["weight"][i, 0] = base_w * tier_w
            # 上下文
            bk["context_cat"][i, :] = ctx_cat
            bk["context_num"][i] = ctx_num
            # 长度
            bk["hist_len"][i] = np.int32(th)
            bk["future_len"][i] = np.int32(tk)

            # 历史
            bk["x_hist"][i, :th, :] = y[:th]
            bk["exo_hist_hour"][i, :th] = exo[:th, 0] + 1
            bk["exo_hist_week"][i, :th] = exo[:th, 1] + 1
            bk["exo_hist_hol"][i, :th]  = exo[:th, 2] + 1

            # 未来
            bk["y_future"][i, :tk, :] = y[t0:]
            bk["future_exo_hour"][i, :tk] = exo[t0:, 0] + 1
            bk["future_exo_week"][i, :tk] = exo[t0:, 1] + 1
            bk["future_exo_hol"][i, :tk]  = exo[t0:, 2] + 1

            # query_time（可选，保留兼容）
            t_since, t_to_end = build_query(t0, T)
            qt = np.concatenate([t_since, t_to_end], axis=-1).astype(np.float32)
            bk["query_time"][i, :tk, :] = qt

    # ---------- 写入 HDF5 ----------
    out_group.attrs["t_min"] = int(t_min)
    out_group.attrs["length_mode"] = "hist_LhLf"
    out_group.attrs["weight_mode"] = str(weight_mode)
    out_group.attrs["weight_floor"] = float(weight_floor)
    out_group.create_dataset("bucket_boundaries", data=np.asarray(bucket_boundaries, np.int32))
    out_group.create_dataset("bucket_counts", data=counts.astype(np.int64))
    out_group.create_dataset("bucket_Lh", data=np.asarray(Lh_list, np.int32))
    out_group.create_dataset("bucket_Lf", data=np.asarray(Lf_list, np.int32))

    for b, bk in enumerate(buckets):
        if bk is None:
            continue
        g = out_group.create_group(f"bucket_{b:02d}")
        g.attrs["n"] = int(bk["n"])
        g.attrs["Lh"] = int(bk["Lh"])
        g.attrs["Lf"] = int(bk["Lf"])
        for k, arr in bk.items():
            if k in ("n", "Lh", "Lf"):
                continue
            g.create_dataset(k, data=arr)