# data/raw.py
import os
from datetime import datetime, timezone
from typing import List, Dict, Optional
from utils.http_utils import get_json_with_retry, save_json
from data.constants import *
from data.cutoff_sources import fetch_and_merge_cutoffs
from data.cutoff_quality import load_cutoff_snapshot
from data.time_features import server_datetime, time_features, TIME_FEATURE_VERSION
from config import Config
import json

_event_data_cache = None
_char_data_cache = None
config = Config()

def _ensure_metadata_loaded():
    global _event_data_cache, _char_data_cache
    if _event_data_cache is None or _char_data_cache is None:
        _event_data_cache, _char_data_cache = _load_local_metadata()

def get_event_data():
    _ensure_metadata_loaded()
    return _event_data_cache

def get_character_data():
    _ensure_metadata_loaded()
    return _char_data_cache

def _cache_metadata(force: bool = False):
    """下载活动与角色元数据，缓存到统一 raw_dir 下（如果已有则跳过）"""
    global _event_data_cache, _char_data_cache
    raw_dir = config.raw_dir
    events_path = os.path.join(raw_dir, "events_all.json")
    chars_path = os.path.join(raw_dir, "characters_main.json")

    if force or not os.path.exists(events_path):
        print("缓存活动元数据...")
        fetched = get_json_with_retry(f"{Bestdoriurl}/api/events/all.6.json")
        if fetched:
            save_json(fetched, events_path)
            _event_data_cache = fetched
            print(f"活动元数据已缓存到 {events_path}")
        elif not os.path.isfile(events_path):
            raise RuntimeError("无法获取活动元数据，且无本地缓存")
    if force or not os.path.exists(chars_path):
        print("缓存角色元数据...")
        fetched = get_json_with_retry(f"{Bestdoriurl}/api/characters/main.3.json")
        if fetched:
            save_json(fetched, chars_path)
            _char_data_cache = fetched
            print(f"角色元数据已缓存到 {chars_path}")
        elif not os.path.isfile(chars_path):
            raise RuntimeError("无法获取角色元数据，且无本地缓存")

def _load_local_metadata():
    """从统一 raw_dir 加载活动与角色数据（JSON 格式）"""
    raw_dir = config.raw_dir
    events_path = os.path.join(raw_dir, "events_all.json")
    chars_path = os.path.join(raw_dir, "characters_main.json")
    if not os.path.exists(events_path) or not os.path.exists(chars_path):
        raise FileNotFoundError(
            "元数据缓存不存在，请先运行下载脚本（download_data.py）或手动缓存"
        )
    with open(events_path, 'r', encoding='utf-8') as f:
        event_data = json.load(f)
    with open(chars_path, 'r', encoding='utf-8') as f:
        char_data = json.load(f)
    return event_data, char_data

def generate_sequences(
    split_threshold: Optional[int] = None,
    event_ids: Optional[List[int]] = None,
    server_list: Optional[List[int]] = None,
    tier_list: Optional[List[int]] = None,
    current_time_ms: Optional[int] = None,
    validation_event_ids: Optional[List[int]] = None,
) -> List[Dict]:
    """
    从统一 raw_dir 生成序列。
    
    - 若 event_ids / server_list / tier_list 未提供，则自动扫描 raw_dir 下所有可用数据。
    - 档位筛选只限制返回结果，保序始终参考同活动所有本地受支持档位。
    - 默认按固定验证 ID 标记 split；完整实验范围与覆盖由 generate_raw_h5.py 检查。
    - 服务可显式传 split_threshold；两种划分参数不能同时使用。
    - 目录结构：config.raw_dir/{server}/{eventId}/{tier}.json
    """
    # 加载本地元数据
    raw_dir = config.raw_dir
    event_data, character_data = _load_local_metadata()
    if split_threshold is not None and validation_event_ids is not None:
        raise ValueError('Cannot combine a fixed validation list with split_threshold')
    if split_threshold is None:
        if validation_event_ids is None:
            from data.event_split import load_event_split
            validation_event_ids = load_event_split(config.split_manifest).validation_ids
        validation_event_ids = frozenset(validation_event_ids)

    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"raw_dir 不存在: {raw_dir}")

    sequences = []
    if current_time_ms is None:
        current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    # 决定遍历方式：提供了过滤列表就用过滤，否则扫描目录
    if event_ids is not None:
        # 按给定参数遍历
        if server_list is None:
            server_list = [3]  # 默认服务器
        for server in server_list:
            for eventId in event_ids:
                event = event_data.get(str(eventId))
                if event is None:
                    print(f"活动 {eventId} 不在元数据中，跳过")
                    continue
                startAt = get_start_at(server, eventId)
                endAt = get_end_at(server, eventId)
                if not startAt or not endAt:
                    continue

                sequences.extend(_create_event_sequences(
                    event, server, eventId, startAt, endAt, character_data,
                    split_threshold, tier_list, current_time_ms,
                    validation_event_ids,
                ))
    else:
        # 原有自动扫描模式
        for server_name in os.listdir(raw_dir):
            server_path = os.path.join(raw_dir, server_name)
            if not os.path.isdir(server_path) or server_name.startswith('.'):
                continue
            try:
                server = int(server_name)
            except ValueError:
                continue
            if server_list is not None and server not in server_list:
                continue

            for event_str in os.listdir(server_path):
                event_path = os.path.join(server_path, event_str)
                if not os.path.isdir(event_path):
                    continue
                try:
                    eventId = int(event_str)
                except ValueError:
                    continue

                event = event_data.get(str(eventId))
                if event is None:
                    print(f"活动 {eventId} 不在元数据中，跳过")
                    continue

                startAt = get_start_at(server, eventId)
                endAt = get_end_at(server, eventId)
                if not startAt or not endAt:
                    continue

                sequences.extend(_create_event_sequences(
                    event, server, eventId, startAt, endAt, character_data,
                    split_threshold, tier_list, current_time_ms,
                    validation_event_ids,
                ))

    return sequences


def _create_event_sequences(
    event, server, event_id, start_at, end_at, character_data,
    split_threshold, requested_tiers, current_time_ms,
    validation_event_ids=None,
):
    """PCHIP -> 同活动累计网格保序 -> 单档增量；训练/服务共用。"""
    raw_by_tier = {}
    for tier in tier_list_for_event(server, event_id):
        path = os.path.join(config.raw_dir, str(server), str(event_id), f"{tier}.json")
        if os.path.isfile(path):
            raw_by_tier[tier] = load_cutoff_snapshot(config.raw_dir, server, event_id, tier)
    if not raw_by_tier:
        return []

    # Missing files are unknown, not structural zeros. Unsupported ranks (e.g.
    # 1500 in old events) are not loaded or used as constraints.
    tiers = sorted(raw_by_tier)
    requested = tiers if requested_tiers is None else list(dict.fromkeys(requested_tiers))
    if not any(tier in raw_by_tier for tier in requested):
        return []
    surface = build_cumulative_grid(raw_by_tier, start_at, end_at, current_time_ms)
    ordered = surface['cutoff']
    columns = {tier: index for index, tier in enumerate(tiers)}
    sequences = []
    for tier in requested:
        if tier not in columns:
            continue
        seq = _create_sequence_from_raw(
            raw_by_tier[tier], event, server, event_id, tier, start_at, end_at,
            character_data, split_threshold, cumulative_ep=ordered[:, columns[tier]],
            validation_event_ids=validation_event_ids,
        )
        sequences.append(seq)
    return sequences


def build_cumulative_grid(raw_by_tier, start_at, end_at, current_time_ms):
    """Shared V1/V2 PCHIP surface. Callers validate raw snapshots first.

    Provenance: 1 synthetic start, 2 exact point, 3 interpolation,
    4 terminal forward fill, 5 confirmed structural zero. Coverage retains
    real timestamps independently of accepted finished-event tail labels.
    """
    import numpy as np
    from data.rank_order import project_ranked_cumulative
    tiers = sorted(raw_by_tier)
    if not tiers:
        raise ValueError("Cannot construct a surface without any cutoff ranks")
    grid = np.arange(start_at, end_at + 1, step_length, dtype=np.int64)
    if len(grid) < 2 or grid[-1] != end_at:
        raise ValueError("Event duration is not a positive multiple of step_length")
    for points in raw_by_tier.values():
        if any(not start_at <= p['time'] <= end_at for p in points):
            raise ValueError("Cutoff points outside event window; refresh raw data")
    cumulative = np.column_stack([
        create_pchip_for_data(raw_by_tier[tier], start_at, end_at)(grid - start_at)
        for tier in tiers
    ])
    limits = np.full(len(tiers), len(grid) - 1, dtype=np.int64)
    latest_times = np.full(len(tiers), -1, dtype=np.int64)
    provenance = np.full(cumulative.shape, 3, dtype=np.uint8)
    provenance[0] = 1
    for index, tier in enumerate(tiers):
        points = raw_by_tier[tier]
        if points:
            latest_times[index] = points[-1]['time']
            provenance[grid > latest_times[index], index] = 4
            provenance[np.isin(grid, [p['time'] for p in points]), index] = 2
        else:
            provenance[:, index] = 5
    if current_time_ms < end_at:
        for index, tier in enumerate(tiers):
            points = raw_by_tier[tier]
            latest = min(latest_times[index], current_time_ms) if points else current_time_ms
            limits[index] = np.clip((latest - start_at) // step_length, -1, len(grid) - 1)
    ordered = project_ranked_cumulative(
        cumulative, limits, zero_tiers=[not raw_by_tier[tier] for tier in tiers],
    )
    return dict(cutoff=ordered, grid_ms=grid, ranks=np.asarray(tiers, dtype=np.int32),
                available_mask=np.arange(len(grid))[:, None] <= limits[None, :],
                provenance=provenance, latest_observation_ms=latest_times,
                rank_repaired_mask=ordered != cumulative)

def get_latest_cutoff_time(event_id: int, server: int, tier: int) -> Optional[int]:
    """返回 cutoff 文件中最近的时间戳（毫秒），若文件不存在或无数据返回 None"""
    raw_dir = config.raw_dir
    file_path = os.path.join(raw_dir, str(server), str(event_id), f"{tier}.json")
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        data = json.load(f)
    if not data:
        return None
    latest = max(item["time"] for item in data)
    return latest

def _create_sequence_from_raw(
    raw_data: list,
    event: dict,
    server: int,
    eventId: int,
    tier: int,
    startAt: int,
    endAt: int,
    character_data: dict,
    split_threshold: Optional[int],
    *,
    cumulative_ep=None,
    validation_event_ids=None,
) -> Optional[Dict]:
    """从原始 cutoff 数据生成单条序列，返回字典或 None（如果数据无效）。"""
    import numpy as np

    if cumulative_ep is None:
        interpolator = create_pchip_for_data(raw_data, startAt, endAt)
        cumulative_ep = interpolator(np.arange(startAt, endAt + 1, step_length) - startAt)
    y_vals = np.diff(cumulative_ep)
    y_arr = np.array(y_vals, dtype=np.float32).reshape(-1, 1)
    exo_arr = time_features(range(startAt + step_length, endAt + 1, step_length), server,
                            get_cn_holidays() if server == 3 else set())

    event_type = eventTypeMap.get(event['eventType'], 0)
    band = get_band(event, character_data)
    tier_idx = model_tier_index(tier)

    # 分别保存分类特征和数值特征
    context_cat = np.array([event_type, band, tier_idx, server, eventId], dtype=np.int32)
    context_num = np.float32(startAt)
    if split_threshold is not None:
        split = "train" if eventId <= split_threshold else "val"
    else:
        split = "val" if eventId in validation_event_ids else "train"
    return {
        "y": y_arr,
        "exo": exo_arr,
        "context_cat": context_cat,
        "context_num": context_num,
        "tier": int(tier),
        "split": split
    }

def download_cutoffs(
    event_list: Optional[List[int]] = None,
    server_list: List[int] = [3],
    tier_list: Optional[List[int]] = None,
    force: bool = False,
    max_workers: int = 1,
):
    """
    下载指定活动、服务器、档位的 cutoff 数据，保存到统一 raw_dir 目录。
    如果文件已存在则跳过（断点续传）。
    """
    out_dir = config.raw_dir
    _cache_metadata()
    event_data, _ = _load_local_metadata()
    tasks = []
    for server in server_list:
        server_events = event_list
        if server_events is None:
            server_events = get_valid_event_ids(server)
        for eventId in server_events:
            event = event_data.get(str(eventId))
            starts = event.get("startAt", []) if event else []
            if server >= len(starts) or starts[server] is None:
                print(f"活动 {eventId} 在服务器 {server} 上不存在，跳过。")
                continue
            event_tiers = tier_list if tier_list is not None else tier_list_for_event(server, eventId)
            for tier in event_tiers:
                if not is_supported_tier(server, eventId, tier):
                    continue
                save_path = os.path.join(out_dir, str(server), str(eventId), f"{tier}.json")
                if not force and os.path.exists(save_path):
                    print(f"已存在，跳过: {save_path}")
                    continue
                start_at = get_start_at(server, eventId)
                end_at = get_end_at(server, eventId)
                tasks.append((server, eventId, tier, start_at, end_at, save_path))

    def download_one(task):
        server, eventId, tier, start_at, end_at, save_path = task
        cutoffs, report = fetch_and_merge_cutoffs(
            server, eventId, tier,
            start_at=start_at,
            end_at=end_at,
        )
        # Failed attempts do not replace the report attached to a saved snapshot.
        attempt_path = os.path.join(
            out_dir, "_fetch_reports", str(server), str(eventId), f"{tier}.json"
        )
        report["fetched_at_ms"] = int(datetime.now(timezone.utc).timestamp() * 1000)
        save_json(report, attempt_path)
        report_path = os.path.join(
            out_dir, "_merge_reports", str(server), str(eventId), f"{tier}.json"
        )
        if not report["successful_sources"]:
            print(f"警告: 所有 cutoff 数据源均获取失败 server={server}, event={eventId}, tier={tier}")
            return
        # 上游短暂返回空数组时，不用空结果破坏已经存在的非空历史缓存。
        if not cutoffs and os.path.exists(save_path):
            try:
                with open(save_path, "r", encoding="utf-8") as existing_file:
                    existing_cutoffs = json.load(existing_file)
            except (OSError, ValueError):
                existing_cutoffs = []
            if existing_cutoffs:
                print(
                    f"警告: 三源并集为空，保留已有非空数据 "
                    f"server={server}, event={eventId}, tier={tier}"
                )
                return
        if not save_json(cutoffs, save_path):
            return
        if not save_json(report, report_path):
            raise OSError(f"Cannot save matching cutoff report: {report_path}")
        source_counts = ", ".join(
            f"{name}={info.get('valid', 0)}"
            for name, info in report["sources"].items()
            if info.get("status") == "ok"
        )
        terminal_drop_removed = sum(
            item.get("removed", 0)
            for item in report["terminal_drop_filter"].values()
        )
        print(
            f"三源合并完成 server={server}, event={eventId}, tier={tier}: "
            f"{source_counts}; 终点下降移除={terminal_drop_removed}; "
            f"StarFX异常移除={report['starfx_filter']['removed']}; "
            f"并集={len(cutoffs)}"
        )

    if max_workers <= 1:
        for task in tasks:
            download_one(task)
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_one, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                server, eventId, tier, *_ = futures[future]
                print(
                    f"警告: cutoff 任务异常 server={server}, event={eventId}, "
                    f"tier={tier}: {type(exc).__name__}: {exc}"
                )

def get_start_at(server: int, eventId: int, *, metadata=None) -> int:
    """根据活动特殊规则调整起始时间（返回毫秒时间戳）"""
    ev = (get_event_data() if metadata is None else metadata).get(str(eventId))
    if ev is None:
        return None
    starts = ev.get('startAt')
    if starts is None or server >= len(starts):
        return None
    timestamp = starts[server]
    if timestamp is None:
        return None
    timestamp = int(timestamp)
    if server == 3:
        if eventId == 293:
            return int(server_datetime(timestamp, server).replace(hour=16, minute=0, second=0, microsecond=0).timestamp() * 1000)
        elif eventId in (312, 313):
            return int(server_datetime(timestamp, server).replace(hour=14, minute=0, second=0, microsecond=0).timestamp() * 1000)
        else:
            return int(server_datetime(timestamp, server).replace(hour=15, minute=0, second=0, microsecond=0).timestamp() * 1000)
    return timestamp

def get_end_at(server: int, eventId: int, *, metadata=None) -> int:
    """根据活动特殊规则调整结束时间"""
    ev = (get_event_data() if metadata is None else metadata).get(str(eventId))
    if ev is None:
        return None
    ends = ev.get('endAt')
    if ends is None or server >= len(ends):
        return None
    timestamp = ends[server]
    if timestamp is None:
        return None
    timestamp = int(timestamp)
    if server == 3 and eventId not in (312, 313):
        return int(server_datetime(timestamp, server).replace(hour=23, minute=0, second=0, microsecond=0).timestamp() * 1000)
    return int(server_datetime(timestamp, server).replace(hour=20, minute=0, second=0, microsecond=0).timestamp() * 1000)

def get_band(event: dict, character_data: dict) -> int:
    """从活动角色推断乐队 ID，返回映射后的索引"""
    bands = set()
    for ch in event.get('characters', []):
        ch_id = str(ch['characterId'])
        if ch_id in character_data:
            bands.add(character_data[ch_id].get('bandId', -1))
    if len(bands) == 1:
        band_id = bands.pop()
    else:
        band_id = 0  # 混合/无法确定
    return bandMap.get(band_id, 0)

def prepare_pchip_points(
    data: List[Dict[str, int]], startAt: int, endAt: int
) -> List[Dict[str, int]]:
    """补齐 PCHIP 边界；非空序列在结束点使用最后 EP 前值延续。"""
    points = [dict(item) for item in sorted(data, key=lambda item: item["time"])]
    if not points:
        return [{"time": startAt, "ep": 0}, {"time": endAt, "ep": 0}]

    if points[0]["time"] > startAt:
        points.insert(0, {"time": startAt, "ep": 0})
    if points[-1]["time"] < endAt:
        points.append({"time": endAt, "ep": points[-1]["ep"]})
    return points


def create_pchip_for_data(data: List[Dict[str, int]], startAt: int, endAt: int):
    """根据 cutoff 数据创建 PCHIP 插值器"""
    import numpy as np
    from scipy.interpolate import PchipInterpolator

    data = prepare_pchip_points(data, startAt, endAt)

    times = np.array([item["time"] - startAt for item in data], dtype=np.float64)
    values = np.array([item["ep"] for item in data], dtype=np.float64)
    sorted_idx = np.argsort(times)
    times = times[sorted_idx]
    values = values[sorted_idx]
    return PchipInterpolator(times, values)

def save_sequences_to_hdf5(sequences: List[Dict], filename: str, *, event_split=None, split_threshold=None):
    import h5py
    import numpy as np
    from data.rank_order import CUTOFF_SEQUENCE_VERSION
    from data.event_split import audit_sequence_split

    if event_split is not None and split_threshold is not None:
        raise ValueError('Cannot combine a fixed event split with split_threshold')
    split_report = audit_sequence_split(sequences, event_split)
    if split_threshold is not None:
        split_report.update(strategy='legacy-id-threshold', version='legacy-id-threshold-v1',
                            split_threshold=int(split_threshold))

    def ensure_2d_y(y):
        if y.ndim == 1:
            return y[:, None]
        return y

    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    with h5py.File(filename, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["y_dim"] = 1
        meta.attrs["exo_dim"] = 3
        meta.attrs["num_sequences"] = len(sequences)
        meta.attrs["cutoff_sequence_version"] = CUTOFF_SEQUENCE_VERSION
        meta.attrs["time_feature_version"] = TIME_FEATURE_VERSION
        meta.attrs["event_split_report"] = json.dumps(split_report, ensure_ascii=False, sort_keys=True)

        for i, seq in enumerate(sequences):
            g = f.create_group(f"seq_{i:06d}")
            g.create_dataset("y", data=ensure_2d_y(seq["y"]), compression="gzip")
            g.create_dataset("exo", data=seq["exo"], compression="gzip")
            g.create_dataset("context_cat", data=seq["context_cat"])
            g.create_dataset("context_num", data=np.float32(seq["context_num"]))
            g.attrs["length"] = len(seq["y"])
            if "tier" in seq:
                g.attrs["tier"] = int(seq["tier"])
            g.attrs["split"] = seq.get("split", "train")


def get_max_event_id() -> int:
    """返回 events_all.json 中所有活动的最大 ID。"""
    event_data, _ = _load_local_metadata()
    if not event_data:
        raise ValueError("events_all.json 中无活动数据")
    return max(int(k) for k in event_data.keys())

def get_active_events(server: int, current_time_ms: Optional[int] = None) -> List[int]:
    """
    返回当前时间（毫秒）下正在进行的活动 ID 列表。
    如果未提供 current_time_ms，则使用当前 UTC 时间。
    """
    if current_time_ms is None:
        current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    event_data, _ = _load_local_metadata()
    active = []
    for ev_id_str, ev in event_data.items():
        ev_id = int(ev_id_str)
        start_at = get_start_at(server, ev_id)
        end_at = get_end_at(server, ev_id)
        if not start_at or not end_at:
            continue
        if start_at <= current_time_ms < end_at:
            active.append(ev_id)
    return sorted(active)

def get_valid_event_ids(server: int, min_event_id: int = 226,
                        current_time_ms: Optional[int] = None) -> List[int]:
    """
    返回所有应处理的活动ID：
    - 活动ID >= min_event_id
    - 活动已经开始（startAt <= current_time_ms）
    """
    if current_time_ms is None:
        current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    event_data, _ = _load_local_metadata()
    valid = []
    for ev_id_str, ev in event_data.items():
        ev_id = int(ev_id_str)
        if ev_id < min_event_id:
            continue
        start_at = get_start_at(server, ev_id)
        if not start_at:
            continue
        if start_at <= current_time_ms:
            valid.append(ev_id)
    return sorted(valid)
