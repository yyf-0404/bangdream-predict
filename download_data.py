# download_data.py
import argparse
from data.raw import download_cutoffs,get_valid_event_ids

def main():
    parser = argparse.ArgumentParser(description="下载原始 cutoff 数据")
    parser.add_argument("--start_event", type=int, default=226)
    parser.add_argument("--out_dir", type=str, default="datasets/raw")
    args = parser.parse_args()

    print(f"准备下载活动 {args.start_event}之后 的 cutoff 数据")
    download_cutoffs(event_list=get_valid_event_ids(server= 3,min_event_id=args.start_event), out_dir=args.out_dir)
    print("下载完成。")

if __name__ == "__main__":
    main()