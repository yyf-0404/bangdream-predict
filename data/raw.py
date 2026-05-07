# data/raw.py
import os
import numpy as np
import h5py
from datetime import datetime, timedelta, timezone
from scipy.interpolate import PchipInterpolator
from typing import List, Dict, Optional
from utils.http_utils import get_json_with_retry, save_json
from data.constants import *
from config import Config
import json

_event_data_cache = None
_char_data_cache = None
config = Config()
def _ensure_metadata_loaded(raw_dir: str):
    global _event_data_cache, _char_data_cache
    if _event_data_cache is None or _char_data_cache is None:
        _event_data_cache, _char_data_cache = _load_local_metadata(raw_dir)
def get_event_data(raw_dir: str = config.raw_dir):
    _ensure_metadata_loaded(raw_dir)
    return _event_data_cache

def get_character_data(raw_dir: str = config.raw_dir):
    _ensure_metadata_loaded(raw_dir)
    return _char_data_cache

def _cache_metadata(raw_dir: str, force: bool = False):
    """下载活动与角色元数据，缓存到 raw_dir 下（如果已有则跳过）"""
    events_path = os.path.join(raw_dir, "events_all.json")
    chars_path = os.path.join(raw_dir, "characters_main.json")

    if force or not os.path.exists(events_path):
        print("缓存活动元数据...")
        event_data = get_json_with_retry(f"{Bestdoriurl}/api/events/all.6.json")
        if event_data:
            save_json(event_data, events_path)
            print(f"活动元数据已缓存到 {events_path}")
        else:
            raise RuntimeError("无法获取活动元数据，且无本地缓存")
    if force or not os.path.exists(chars_path):
        print("缓存角色元数据...")
        char_data = get_json_with_retry(f"{Bestdoriurl}/api/characters/main.3.json")
        if char_data:
            save_json(char_data, chars_path)
            print(f"角色元数据已缓存到 {chars_path}")
        else:
            raise RuntimeError("无法获取角色元数据，且无本地缓存")

def _load_local_metadata(raw_dir: str):
    """从 raw_dir 加载活动与角色数据（JSON 格式）"""
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
    raw_dir: str = "raw",
    split_threshold: int = 280,
    event_ids: Optional[List[int]] = None,
    server_list: Optional[List[int]] = None,
    tier_list: Optional[List[int]] = None,
) -> List[Dict]:
    """
    从 raw_dir 生成序列。
    
    - 若 event_ids / server_list / tier_list 未提供，则自动扫描 raw_dir 下所有可用数据。
    - 若提供，则只生成匹配指定活动ID、服务器、档位的序列。
    - 目录结构：raw_dir/{server}/{eventId}/{tier}.json
    """
    # 加载本地元数据
    event_data, character_data = _load_local_metadata(raw_dir)

    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"raw_dir 不存在: {raw_dir}")

    sequences = []

    # 决定遍历方式：提供了过滤列表就用过滤，否则扫描目录
    if event_ids is not None:
        # 按给定参数遍历
        if server_list is None:
            server_list = [3]  # 默认服务器
        for server in server_list:
            if tier_list is None:
                tier_list = tierListOfServer[server]  # 默认使用指定服务器的所有档位
            for eventId in event_ids:
                event = event_data.get(str(eventId))
                if event is None:
                    print(f"活动 {eventId} 不在元数据中，跳过")
                    continue
                startAt = get_start_at(server, eventId)
                endAt = get_end_at(server, eventId)
                if not startAt or not endAt:
                    continue

                for tier in tier_list:
                    if tier not in tierListOfServer.get(server, []):
                        continue
                    file_path = os.path.join(raw_dir, str(server), str(eventId), f"{tier}.json")
                    if not os.path.exists(file_path):
                        print(f"警告: 缺少 cutoff 文件 {file_path}，跳过")
                        continue
                    with open(file_path, 'r') as f:
                        raw_data = json.load(f)

                    # 生成序列（与原有逻辑相同）
                    seq = _create_sequence_from_raw(
                        raw_data, event, server, eventId, tier,
                        startAt, endAt, character_data, split_threshold
                    )
                    if seq:
                        sequences.append(seq)
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

                for file_name in os.listdir(event_path):
                    if not file_name.endswith('.json'):
                        continue
                    tier_str = file_name[:-5]
                    try:
                        tier = int(tier_str)
                    except ValueError:
                        print(f"非整数 tier 文件名，跳过: {file_name}")
                        continue

                    file_path = os.path.join(event_path, file_name)
                    with open(file_path, 'r') as f:
                        raw_data = json.load(f)

                    seq = _create_sequence_from_raw(
                        raw_data, event, server, eventId, tier,
                        startAt, endAt, character_data, split_threshold
                    )
                    if seq:
                        sequences.append(seq)

    return sequences

def get_latest_cutoff_time(event_id: int, server: int, tier: int, raw_dir: str = None) -> Optional[int]:
    """返回 cutoff 文件中最近的时间戳（毫秒），若文件不存在或无数据返回 None"""
    if raw_dir is None:
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
    split_threshold: int,
) -> Optional[Dict]:
    """从原始 cutoff 数据生成单条序列，返回字典或 None（如果数据无效）。"""
    interpolator = create_pchip_for_data(raw_data, startAt, endAt)
    y_vals = []
    exo_vals = []
    for t in range(startAt, endAt, step_length):
        val = max(interpolator(t + step_length - startAt) - interpolator(t - startAt), 0.0)
        y_vals.append(val)
        date = datetime.fromtimestamp((t + step_length) // 1000)
        tomorrow = date + timedelta(days=1)

        today_hol = date in cn_holidays
        tom_hol = tomorrow in cn_holidays

        if not today_hol and not tom_hol:
            day_type = 0
        elif not today_hol and tom_hol:
            day_type = 1
        elif today_hol and not tom_hol:
            day_type = 3
        else:
            day_type = 2

        exo_vals.append([
            date.hour * 2 + date.minute // 30,
            date.weekday(),
            day_type
        ])

    y_arr = np.array(y_vals, dtype=np.float32).reshape(-1, 1)
    exo_arr = np.array(exo_vals, dtype=np.int32)

    event_type = eventTypeMap.get(event['eventType'], 0)
    band = get_band(event, character_data)
    tier_idx = tierMap.get(tier)
    if tier_idx is None:
        print(f"警告: tier {tier} 不在 tierMap 中，跳过")
        return None

    # 分别保存分类特征和数值特征
    context_cat = np.array([event_type, band, tier_idx, server, eventId], dtype=np.int32)
    context_num = np.float32(startAt)
    split = "train" if eventId <= split_threshold else "val"
    return {
        "y": y_arr,
        "exo": exo_arr,
        "context_cat": context_cat,
        "context_num": context_num,
        "split": split
    }

def download_cutoffs(
    event_list: Optional[List[int]] = None,
    server_list: List[int] = [3],
    tier_list: Optional[List[int]] = None,
    out_dir: str = config.raw_dir,
    force: bool = False,
):
    """
    下载指定活动、服务器、档位的 cutoff 数据，保存到 raw/ 目录。
    如果文件已存在则跳过（断点续传）。
    """
    if tier_list is None:
        tier_list = tierListOfServer[3]  # 默认使用所有档位

    _cache_metadata(out_dir)
    event_data, _ = _load_local_metadata(out_dir)
    for server in server_list:
        for eventId in event_list:
            if str(eventId) not in event_data or event_data[str(eventId)]['startAt'][server] is None:
                print(f"活动 {eventId} 在服务器 {server} 上不存在，跳过。")
                continue
            for tier in tier_list:
                save_path = os.path.join(out_dir, str(server), str(eventId), f"{tier}.json")
                if not force and os.path.exists(save_path):
                    print(f"已存在，跳过: {save_path}")
                    continue
                url = f"{Bestdoriurl}/api/tracker/data?server={server}&event={eventId}&tier={tier}"
                data = get_json_with_retry(url)
                if data is None:
                    print(f"警告: 无法获取 server={server}, event={eventId}, tier={tier}")
                    continue
                cutoffs = data.get("cutoffs", [])
                save_json(cutoffs, save_path)

def get_start_at(server: int, eventId: int) -> int:
    """根据活动特殊规则调整起始时间（返回毫秒时间戳）"""
    ev = get_event_data().get(str(eventId))
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
            return int(datetime.fromtimestamp(timestamp // 1000).replace(hour=16, minute=0, second=0).timestamp() * 1000)
        elif eventId in (312, 313):
            return int(datetime.fromtimestamp(timestamp // 1000).replace(hour=14, minute=0, second=0).timestamp() * 1000)
        else:
            return int(datetime.fromtimestamp(timestamp // 1000).replace(hour=15, minute=0, second=0).timestamp() * 1000)
    return timestamp

def get_end_at(server: int, eventId: int) -> int:
    """根据活动特殊规则调整结束时间"""
    ev = get_event_data().get(str(eventId))
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
        return int(datetime.fromtimestamp(timestamp // 1000).replace(hour=23, minute=0, second=0).timestamp() * 1000)
    return int(datetime.fromtimestamp(timestamp // 1000).replace(hour=20, minute=0, second=0).timestamp() * 1000)

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

def create_pchip_for_data(data: List[Dict[str, int]], startAt: int, endAt: int) -> PchipInterpolator:
    """根据 cutoff 数据创建 PCHIP 插值器"""
    if len(data) == 0:
        data = [{"time": endAt, "ep": 0}]
    # 在两端补点（startAt, 0）和（endAt, 0）
    data.insert(0, {"time": startAt, "ep": 0})
    if data[-1]["time"] != endAt:
        data.append({"time": endAt, "ep": 0})

    times = np.array([item["time"] - startAt for item in data], dtype=np.float64)
    values = np.array([item["ep"] for item in data], dtype=np.float64)
    sorted_idx = np.argsort(times)
    times = times[sorted_idx]
    values = values[sorted_idx]
    return PchipInterpolator(times, values)

def save_sequences_to_hdf5(sequences: List[Dict], filename: str):
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

        for i, seq in enumerate(sequences):
            g = f.create_group(f"seq_{i:06d}")
            g.create_dataset("y", data=ensure_2d_y(seq["y"]), compression="gzip")
            g.create_dataset("exo", data=seq["exo"], compression="gzip")
            g.create_dataset("context_cat", data=seq["context_cat"])
            g.create_dataset("context_num", data=np.float32(seq["context_num"]))
            g.attrs["length"] = len(seq["y"])
            g.attrs["split"] = seq.get("split", "train")


def get_max_event_id(raw_dir: str) -> int:
    """返回 events_all.json 中所有活动的最大 ID。"""
    event_data, _ = _load_local_metadata(raw_dir)
    if not event_data:
        raise ValueError("events_all.json 中无活动数据")
    return max(int(k) for k in event_data.keys())

def get_active_events(server: int, raw_dir: str = config.raw_dir,
                      current_time_ms: Optional[int] = None) -> List[int]:
    """
    返回当前时间（毫秒）下正在进行的活动 ID 列表。
    如果未提供 current_time_ms，则使用当前 UTC 时间。
    """
    if current_time_ms is None:
        current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    event_data, _ = _load_local_metadata(raw_dir)
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

def get_valid_event_ids(server: int, raw_dir: str = config.raw_dir,
                        min_event_id: int = 226,
                        current_time_ms: Optional[int] = None) -> List[int]:
    """
    返回所有应处理的活动ID：
    - 活动ID >= min_event_id
    - 活动已经开始（startAt <= current_time_ms）
    """
    if current_time_ms is None:
        current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    event_data, _ = _load_local_metadata(raw_dir)
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