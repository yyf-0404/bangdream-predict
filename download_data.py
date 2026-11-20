# download_data.py
import argparse
from data.raw import _cache_metadata, download_cutoffs, get_valid_event_ids

def main():
    parser = argparse.ArgumentParser(description="下载原始 cutoff 数据")
    parser.add_argument("--start_event", type=int, default=226)
    parser.add_argument("--force", action="store_true", help="重新获取并覆盖已有的合并 cutoff")
    parser.add_argument("--workers", type=int, default=4, help="并行处理的活动档位数")
    args = parser.parse_args()

    # 全新环境中必须先下载元数据；完整强制刷新时也同步刷新活动范围。
    _cache_metadata(force=args.force)
    print(f"准备下载活动 {args.start_event}之后 的 cutoff 数据")
    download_cutoffs(
        event_list=get_valid_event_ids(server=3, min_event_id=args.start_event),
        force=args.force,
        max_workers=max(1, args.workers),
    )
    print("下载完成。")

if __name__ == "__main__":
    main()
