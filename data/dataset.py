# data/dataset.py
import h5py
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

@dataclass
class BucketData:
    name: str
    n: int
    Lh: int
    Lf: int
    arrays: Dict[str, np.ndarray]

def load_preprocessed(pre_h5: str, split: str) -> Tuple[Dict[str, Any], List[BucketData]]:
    """从预处理 HDF5 加载指定 split 的桶数据。"""
    import re
    _BUCKET_RE = re.compile(r"^bucket_\d+$")

    with h5py.File(pre_h5, "r") as f:
        meta = f["meta"].attrs
        x_dim = int(meta["x_dim"])
        context_cat_len = int(meta["context_cat_len"])

        y_mean = f["norm/y_mean"][:].astype(np.float32)
        y_std = f["norm/y_std"][:].astype(np.float32)
        ctx_min = float(f["norm/ctx_min"][0])
        ctx_max = float(f["norm/ctx_max"][0])

        split_g = f[split]
        bucket_keys = []
        for k in split_g.keys():
            if not _BUCKET_RE.match(k):
                continue
            obj = split_g[k]
            if isinstance(obj, h5py.Group) and ("x_hist" in obj):
                bucket_keys.append(k)
        bucket_keys.sort()

        buckets: List[BucketData] = []
        for bk_name in bucket_keys:
            gb = split_g[bk_name]
            n = int(gb.attrs.get("n", gb["x_hist"].shape[0]))
            Lh = int(gb.attrs.get("Lh", gb["x_hist"].shape[1]))
            Lf = int(gb.attrs.get("Lf", gb["y_future"].shape[1]))

            arrays: Dict[str, np.ndarray] = {}
            keys = [
                "x_hist",
                "exo_hist_hour", "exo_hist_week", "exo_hist_hol",
                "hist_len",
                "context_cat", "context_num",
                "future_exo_hour", "future_exo_week", "future_exo_hol",
                "future_len",
                "y_future",
                "weight",
            ]
            for kk in keys:
                if kk not in gb:
                    raise KeyError(f"Bucket {bk_name} missing dataset: {kk}")
                arrays[kk] = gb[kk][:]
            buckets.append(BucketData(name=bk_name, n=n, Lh=Lh, Lf=Lf, arrays=arrays))

    norm = dict(
        x_dim=x_dim,
        context_cat_len=context_cat_len,
        y_mean=y_mean,
        y_std=y_std,
        ctx_min=ctx_min,
        ctx_max=ctx_max,
    )
    return norm, buckets

def steps_from_buckets(buckets: List[BucketData], batch_size: int) -> int:
    return int(sum((b.n // batch_size) for b in buckets))

def make_bucket_dataset(bucket: BucketData, batch_size: int, training: bool, seed: int) -> tf.data.Dataset:
    a = bucket.arrays
    x = {
        "x_hist": a["x_hist"].astype(np.float32),
        "exo_hist_hour": a["exo_hist_hour"].astype(np.int32),
        "exo_hist_week": a["exo_hist_week"].astype(np.int32),
        "exo_hist_hol": a["exo_hist_hol"].astype(np.int32),
        "hist_len": a["hist_len"].astype(np.int32),
        "context_cat": a["context_cat"].astype(np.int32),
        "context_num": a["context_num"].astype(np.float32),
        "future_exo_hour": a["future_exo_hour"].astype(np.int32),
        "future_exo_week": a["future_exo_week"].astype(np.int32),
        "future_exo_hol": a["future_exo_hol"].astype(np.int32),
        "future_len": a["future_len"].astype(np.int32),
    }
    y = a["y_future"].astype(np.float32)
    w = a["weight"].astype(np.float32).reshape(-1)

    ds = tf.data.Dataset.from_tensor_slices((x, y, w))
    if training:
        ds = ds.shuffle(buffer_size=min(bucket.n, 8192), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)   # 保持静态 batch shape
    return ds

def make_batch_dataset_from_buckets(
    buckets: List[BucketData],
    batch_size: int,
    training: bool,
    seed: int,
    repeat: bool,
) -> Tuple[tf.data.Dataset, int]:

    steps = steps_from_buckets(buckets, batch_size)
    if steps <= 0:
        raise ValueError("No full batches available. Reduce batch_size or add more data.")

    per_bucket_ds = []
    weights = []
    for i, b in enumerate(buckets):
        if b.n < batch_size:
            continue
        per_bucket_ds.append(make_bucket_dataset(b, batch_size, training=training, seed=seed + i * 1000))
        weights.append(float(b.n))

    if not per_bucket_ds:
        raise ValueError("No buckets have enough samples for one batch.")

    if training:
        ds = tf.data.Dataset.sample_from_datasets(per_bucket_ds, weights=weights, seed=seed)
        if repeat:
            ds = ds.repeat()
    else:
        ds = per_bucket_ds[0]
        for d in per_bucket_ds[1:]:
            ds = ds.concatenate(d)

    options = tf.data.Options()
    options.autotune.enabled = True
    ds = ds.with_options(options)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, steps