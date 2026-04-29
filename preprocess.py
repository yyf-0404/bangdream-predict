# preprocess.py
import os, argparse, h5py
import numpy as np
from datetime import datetime
from config import Config
from data.sequences import load_sequences_from_hdf5, compute_y_normalization_params, compute_contextnum_params
from data.slicing import choose_t0_lists
from data.preprocessing import preprocess_split_to_groups

def main():
    parser = argparse.ArgumentParser(description="预处理：从 data.hdf5 生成 preprocessed.h5")
    parser.add_argument("--raw_h5", type=str, default=None)
    parser.add_argument("--out_h5", type=str, default=None)
    parser.add_argument("--t_min", type=int, default=None)
    parser.add_argument("--random_slices", type=int, default=None)
    parser.add_argument("--weight_mode", type=str, default=None)
    parser.add_argument("--weight_floor", type=float, default=None)
    parser.add_argument("--boundaries", nargs="*", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = Config()
    # 命令行覆盖
    if args.raw_h5 is not None: config.raw_h5 = args.raw_h5
    out_h5 = args.out_h5 if args.out_h5 is not None else config.pre_h5
    if args.t_min is not None: config.t_min = args.t_min
    if args.random_slices is not None: config.random_slices_per_seq = args.random_slices
    if args.weight_mode is not None: config.weight_mode = args.weight_mode
    if args.weight_floor is not None: config.weight_floor = args.weight_floor
    if args.boundaries is not None: config.bucket_boundaries = args.boundaries
    if args.seed is not None: config.seed = args.seed

    if not os.path.exists(config.raw_h5):
        raise FileNotFoundError(f"原始数据文件不存在: {config.raw_h5}")

    # 加载序列
    train_seqs = load_sequences_from_hdf5(config.raw_h5, "train")
    val_seqs   = load_sequences_from_hdf5(config.raw_h5, "val")
    if not train_seqs or not val_seqs:
        raise RuntimeError("train/val 序列为空，请检查 split 标记")

    x_dim = train_seqs[0]["y"].shape[1]
    context_cat_len = train_seqs[0]["context_cat"].shape[0]

    # 全局统计
    y_mean, y_std = compute_y_normalization_params(train_seqs)
    ctx_min, ctx_max = compute_contextnum_params(train_seqs)

    # 生成切点列表（固定种子）
    train_t0 = choose_t0_lists(train_seqs, config.t_min, config.random_slices_per_seq, config.seed)
    val_t0   = choose_t0_lists(val_seqs,   config.t_min, config.random_slices_per_seq, config.seed + 12345)

    # 写文件（临时文件原子替换）
    tmp = out_h5 + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)

    with h5py.File(tmp, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["created_at"] = datetime.now().isoformat()
        meta.attrs["raw_source"] = config.raw_h5
        meta.attrs["x_dim"] = int(x_dim)
        meta.attrs["context_cat_len"] = int(context_cat_len)
        meta.attrs["t_min"] = int(config.t_min)
        meta.attrs["seed"] = int(config.seed)
        meta.attrs["random_slices_per_seq"] = -1 if config.random_slices_per_seq is None else int(config.random_slices_per_seq)
        meta.attrs["weight_mode"] = str(config.weight_mode)
        meta.attrs["weight_floor"] = float(config.weight_floor)

        norm = f.create_group("norm")
        norm.create_dataset("y_mean", data=y_mean)
        norm.create_dataset("y_std", data=y_std)
        norm.create_dataset("ctx_min", data=np.asarray([ctx_min], np.float32))
        norm.create_dataset("ctx_max", data=np.asarray([ctx_max], np.float32))

        gtr = f.create_group("train")
        gva = f.create_group("val")

        preprocess_split_to_groups(
            train_seqs, train_t0, gtr,
            t_min=config.t_min, x_dim=x_dim, context_cat_len=context_cat_len,
            bucket_boundaries=config.bucket_boundaries,
            length_mode=config.length_mode,
            weight_mode=config.weight_mode, weight_floor=config.weight_floor,
            tier_center=config.tier_center, tier_alpha=config.tier_alpha, tier_floor=config.tier_floor,
        )
        preprocess_split_to_groups(
            val_seqs, val_t0, gva,
            t_min=config.t_min, x_dim=x_dim, context_cat_len=context_cat_len,
            bucket_boundaries=config.bucket_boundaries,
            length_mode=config.length_mode,
            weight_mode=config.weight_mode, weight_floor=config.weight_floor,
            tier_center=config.tier_center, tier_alpha=config.tier_alpha, tier_floor=config.tier_floor,
        )

    os.replace(tmp, out_h5)
    print(f"预处理完成，输出: {out_h5}")

if __name__ == "__main__":
    main()