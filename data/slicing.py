# data/slicing.py
import numpy as np
from typing import List, Dict, Any

def build_query(t0: int, T: int):
    """未来每个步的 t_since 和 t_to_end 比例。"""
    K = T - t0
    t_since = np.arange(1, K + 1, dtype=np.float32) / float(T)
    t_to_end = np.arange(K, 0, -1, dtype=np.float32) / float(T)
    return t_since.reshape(-1, 1), t_to_end.reshape(-1, 1)

def choose_t0_lists(
    sequences: List[Dict[str, Any]],
    t_min: int,
    random_slices_per_seq: int = None,
    seed: int = 1
) -> List[List[int]]:
    """为每条序列生成切点 t0 列表。"""
    rng = np.random.default_rng(seed)
    t0_lists = []
    for seq in sequences:
        T = len(seq["y"])
        if T <= t_min:
            t0_lists.append([])
            continue
        candidates = np.arange(t_min, T, dtype=np.int32)
        if random_slices_per_seq is None:
            t0_list = candidates.tolist()
        else:
            n_req = int(random_slices_per_seq)
            if n_req <= 0:
                t0_list = []
            else:
                n_pick = min(n_req, len(candidates))
                if n_pick == len(candidates):
                    t0_list = candidates.tolist()
                else:
                    t0_list = rng.choice(candidates, size=n_pick, replace=False).tolist()
                    t0_list.sort()
        t0_lists.append(t0_list)
    return t0_lists

def _build_inputs_for_slices_dict(seq: Dict[str, Any], t0_list: List[int]) -> Dict[str, np.ndarray]:
    """
    根据切片点列表构造模型输入字典（与训练数据格式一致）。
    seq: 单个序列字典（来自 load_sequences_from_hdf5）
    t0_list: 切点 t0 的列表
    返回: x 字典，格式与训练时相同
    """
    y = seq["y"].astype(np.float32)                 # (T, x_dim) 通常为 (T,1)
    exo = seq["exo"].astype(np.int32)               # (T, 3) 0-based
    # context_cat 原为 0-based，转为 1-based（0 留给 padding）
    context_cat = (seq["context_cat"].astype(np.int32) + 1)
    context_num = np.float32(seq["context_num"])
    T, x_dim = y.shape

    B = len(t0_list)
    if B == 0:
        return {
            "x_hist": np.zeros((0, 0, x_dim), np.float32),
            "exo_hist_hour": np.zeros((0, 0), np.int32),
            "exo_hist_week": np.zeros((0, 0), np.int32),
            "exo_hist_hol": np.zeros((0, 0), np.int32),
            "hist_len": np.zeros((0,), np.int32),
            "context_cat": np.zeros((0, 4), np.int32),
            "context_num": np.zeros((0,), np.float32),
            "future_exo_hour": np.zeros((0, 0), np.int32),
            "future_exo_week": np.zeros((0, 0), np.int32),
            "future_exo_hol": np.zeros((0, 0), np.int32),
            "future_len": np.zeros((0,), np.int32),
            "weight": np.ones((0, 1), np.float32),
        }

    th_list = np.array(t0_list, dtype=np.int32)
    k_list = (T - th_list).astype(np.int32)

    max_th = int(th_list.max())
    max_k  = int(k_list.max())

    x_hist = np.zeros((B, max_th, x_dim), np.float32)
    exo_hist_hour = np.zeros((B, max_th), np.int32)
    exo_hist_week = np.zeros((B, max_th), np.int32)
    exo_hist_hol  = np.zeros((B, max_th), np.int32)

    future_exo_hour = np.zeros((B, max_k), np.int32)
    future_exo_week = np.zeros((B, max_k), np.int32)
    future_exo_hol  = np.zeros((B, max_k), np.int32)

    context_cat_in = np.tile(context_cat[None, :], (B, 1)).astype(np.int32)
    context_num_in = np.full((B,), context_num, dtype=np.float32)

    for i, t0 in enumerate(t0_list):
        th = int(t0)
        k  = int(T - t0)

        x_hist[i, :th, :] = y[:th, :]
        exo_hist_hour[i, :th] = exo[:th, 0] + 1
        exo_hist_week[i, :th] = exo[:th, 1] + 1
        exo_hist_hol[i, :th]  = exo[:th, 2] + 1

        future_exo_hour[i, :k] = exo[t0:, 0] + 1
        future_exo_week[i, :k] = exo[t0:, 1] + 1
        future_exo_hol[i, :k]  = exo[t0:, 2] + 1

    return {
        "x_hist": x_hist,
        "exo_hist_hour": exo_hist_hour,
        "exo_hist_week": exo_hist_week,
        "exo_hist_hol": exo_hist_hol,
        "hist_len": th_list,
        "context_cat": context_cat_in,
        "context_num": context_num_in,
        "future_exo_hour": future_exo_hour,
        "future_exo_week": future_exo_week,
        "future_exo_hol": future_exo_hol,
        "future_len": k_list,
        "weight": np.ones((B, 1), np.float32),
    }
