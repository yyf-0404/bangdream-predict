# generate_raw_h5.py
import argparse
from config import Config
from data.raw import generate_sequences, save_sequences_to_hdf5, get_valid_event_ids, get_active_events

def main():
    parser = argparse.ArgumentParser(description="从 raw 目录生成 data.hdf5（自动扫描所有可用数据）")
    parser.add_argument("--out_h5", type=str, default=None, help="输出 HDF5 路径（默认 config.raw_h5）")
    parser.add_argument("--split_threshold", type=int, default=None,
                        help="训练/验证划分的活动ID阈值（默认 config.split_threshold）")
    args = parser.parse_args()

    config = Config()
    out_h5 = args.out_h5 if args.out_h5 is not None else config.raw_h5
    split_threshold = args.split_threshold if args.split_threshold is not None else config.split_threshold
    event_ids = list(set(get_valid_event_ids(3)) - set(get_active_events(3)))
    print(f"生成序列范围为{min(event_ids)}-{max(event_ids)}共 {len(event_ids)} 个活动 (split 阈值 = {split_threshold})...")
    seqs = generate_sequences(event_ids=event_ids, split_threshold=split_threshold)
    save_sequences_to_hdf5(seqs, out_h5)
    print(f"完成：共 {len(seqs)} 条序列，保存至 {out_h5}")
    
if __name__ == "__main__":
    main()
