# data/sequences.py
import h5py
import numpy as np
from typing import List, Dict, Any, Optional

def load_sequences_from_hdf5(filename: str, split: Optional[str] = None) -> List[Dict[str, Any]]:
    sequences = []
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            if not key.startswith("seq_"):
                continue
            g = f[key]
            if split is not None:
                split_attr = g.attrs.get("split", None)
                if isinstance(split_attr, bytes):
                    split_attr = split_attr.decode("utf-8")
                if split_attr != split:
                    continue

            y = g["y"][:].astype(np.float32)
            exo = g["exo"][:].astype(np.int32)
            context_cat = g["context_cat"][:].astype(np.int32)
            context_num = np.float32(g["context_num"][()])

            sequences.append({
                "y": y,
                "exo": exo,
                "context_cat": context_cat,
                "context_num": context_num,
            })
    return sequences

def compute_y_normalization_params(sequences: List[Dict[str, Any]]):
    """对训练集计算 log1p(y) 的均值与标准差。"""
    all_y = np.concatenate([seq["y"] for seq in sequences], axis=0)
    all_y = np.maximum(all_y, 0.0)
    all_y_log = np.log1p(all_y)
    y_mean = all_y_log.mean(axis=0)
    y_std = all_y_log.std(axis=0) + 1e-8
    return y_mean.astype(np.float32), y_std.astype(np.float32)

def compute_contextnum_params(sequences: List[Dict[str, Any]]):
    """数值上下界的扩展（0.5 外扩）。"""
    all_ctx = np.array([seq["context_num"] for seq in sequences], dtype=np.float32)
    min_val = float(all_ctx.min())
    max_val = float(all_ctx.max())
    extension = 0.5 * (max_val - min_val)
    return float(min_val), float(max_val + extension)