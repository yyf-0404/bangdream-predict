"""多数据源 cutoff 获取、清洗与合并。

活动档线是累计值，因此合法序列必须同时满足：时间有效、EP 非负、
按时间排序后 EP 不下降。StarFX 在此基础上还会使用 Bestdori/HHWX
组成的可信参考包络进行异常检测。
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlencode
from .cutoff_quality import QUALITY_VERSION, cutoff_digest


BESTDORI_BASE_URL = "https://bestdori.com"
STARFX_BASE_URL = os.environ.get(
    "RANKING_API_BASE_URL", "https://grp-speed-backend.starfreedomx.top"
).rstrip("/")
HHWX_BASE_URL = os.environ.get("HHWX_API_BASE_URL", "https://hhwx.org").rstrip("/")

# 同一时间戳发生冲突时优先采用较可信的数据源。该优先级不影响并集中的
# 非冲突时间点。
SOURCE_PRIORITY = {"bestdori": 0, "hhwx": 1, "starfx": 2}
DEFAULT_STARFX_RELATIVE_TOLERANCE = 0.03
DEFAULT_STARFX_ABSOLUTE_TOLERANCE = 5000


def cutoff_source_urls(server: int, event_id: int, tier: int) -> Dict[str, str]:
    """返回指定服务器可用的 tracker 数据源。

    与参考项目保持一致：CN 使用三源，JP 使用 StarFX + Bestdori，
    其他服务器仅使用 Bestdori。
    """
    query = urlencode({"server": int(server), "event": int(event_id), "tier": int(tier)})
    sources = {
        "bestdori": f"{BESTDORI_BASE_URL}/api/tracker/data?{query}",
    }
    if server in (0, 3):
        sources["starfx"] = f"{STARFX_BASE_URL}/api/tracker/data?{query}"
    if server == 3:
        sources["hhwx"] = f"{HHWX_BASE_URL}/api/bandori/tracker/data?{query}"
    return sources


def _coerce_timestamp_ms(value) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    # 对兼容数据源返回的秒级 Unix 时间戳进行统一。
    if number < 100_000_000_000:
        number *= 1000.0
    return int(round(number))


def _coerce_ep(value) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return int(round(number))


def _extract_cutoffs(payload) -> Optional[list]:
    """解析三个兼容 tracker 接口的响应；None 表示响应无效。"""
    if isinstance(payload, dict):
        if payload.get("result") is False:
            return None
        cutoffs = payload.get("cutoffs")
        return cutoffs if isinstance(cutoffs, list) else None
    # 兼容测试、离线导入以及直接返回数组的镜像。
    return payload if isinstance(payload, list) else None


def normalize_cutoff_points(
    raw_points: Sequence[dict],
    *,
    start_at: Optional[int] = None,
    end_at: Optional[int] = None,
) -> Tuple[List[dict], dict]:
    """统一字段类型、过滤非法值，并解决单源内部的重复时间戳。"""
    by_time: Dict[int, List[int]] = defaultdict(list)
    invalid = 0
    outside_window = 0

    for item in raw_points:
        if not isinstance(item, dict):
            invalid += 1
            continue
        timestamp = _coerce_timestamp_ms(item.get("time", item.get("timestamp")))
        ep = _coerce_ep(item.get("ep", item.get("value")))
        if timestamp is None or ep is None:
            invalid += 1
            continue
        if start_at is not None and timestamp < int(start_at):
            outside_window += 1
            continue
        if end_at is not None and timestamp > int(end_at):
            outside_window += 1
            continue
        by_time[timestamp].append(ep)

    points = []
    duplicate_time = 0
    for timestamp in sorted(by_time):
        values = sorted(by_time[timestamp])
        duplicate_time += max(0, len(values) - 1)
        # 选择实际观测中的下中位数，不凭空合成新的 EP。
        ep = values[(len(values) - 1) // 2]
        points.append({"time": timestamp, "ep": ep})

    return points, {
        "received": len(raw_points),
        "valid": len(points),
        "invalid": invalid,
        "outside_event_window": outside_window,
        "duplicate_time": duplicate_time,
    }


def filter_terminal_drop(
    points: Sequence[dict], end_at: Optional[int]
) -> Tuple[List[dict], dict]:
    """忽略精确 endAt 相对最后赛中观测下降的结算点。

    该步骤必须发生在单调清洗之前，否则最后一个合法赛中高点可能被误判为
    孤立尖峰。正常上涨或持平的 endAt 点保持不变。
    """
    working = [dict(point) for point in sorted(points, key=lambda point: point["time"])]
    report = {"removed": 0, "rejected": []}
    if end_at is None or len(working) < 2 or working[-1]["time"] != int(end_at):
        return working, report

    endpoint = working[-1]
    previous = working[-2]
    if endpoint["ep"] >= previous["ep"]:
        return working, report

    rejected = {
        **endpoint,
        "previous_time": previous["time"],
        "previous_ep": previous["ep"],
        "reason": "terminal_settlement_drop",
    }
    return working[:-1], {"removed": 1, "rejected": [rejected]}


def _longest_nondecreasing_indices(values: Sequence[int]) -> set[int]:
    """返回一个最长非下降子序列的索引集合，避免单个高尖峰污染其后数据。"""
    if not values:
        return set()

    tails: List[int] = []
    tail_indices: List[int] = []
    previous = [-1] * len(values)

    for index, value in enumerate(values):
        position = bisect_right(tails, value)
        if position == len(tails):
            tails.append(value)
            tail_indices.append(index)
        else:
            tails[position] = value
            tail_indices[position] = index
        if position > 0:
            previous[index] = tail_indices[position - 1]

    keep = set()
    cursor = tail_indices[-1]
    while cursor >= 0:
        keep.add(cursor)
        cursor = previous[cursor]
    return keep


def _clean_monotonic_points(points: Sequence[dict]) -> Tuple[List[dict], List[dict]]:
    """去除累计序列中的孤立尖峰/低谷，并保留最长非下降子序列。"""
    working = [dict(point) for point in sorted(points, key=lambda point: point["time"])]
    rejected: List[dict] = []

    # 先删除两侧可信且有序时明显落在区间外的孤立点，避免 LNDS 在等长方案中
    # 选择低谷而丢掉正常点。
    changed = True
    while changed and len(working) >= 3:
        changed = False
        for index in range(1, len(working) - 1):
            prev_ep = working[index - 1]["ep"]
            ep = working[index]["ep"]
            next_ep = working[index + 1]["ep"]
            if prev_ep <= next_ep and not (prev_ep <= ep <= next_ep):
                point = working.pop(index)
                rejected.append({**point, "reason": "isolated_monotonic_violation"})
                changed = True
                break

    keep_indices = _longest_nondecreasing_indices([point["ep"] for point in working])
    kept = []
    for index, point in enumerate(working):
        if index in keep_indices:
            kept.append(point)
        else:
            rejected.append({**point, "reason": "monotonic_violation"})
    return kept, rejected


def filter_starfx_outliers(
    starfx_points: Sequence[dict],
    trusted_reference: Sequence[dict],
    *,
    relative_tolerance: float = DEFAULT_STARFX_RELATIVE_TOLERANCE,
    absolute_tolerance: int = DEFAULT_STARFX_ABSOLUTE_TOLERANCE,
) -> Tuple[List[dict], dict]:
    """过滤 StarFX 异常值。

    第一层使用累计档线的单调性；第二层使用 Bestdori/HHWX 的前后观测构造
    参考包络。参考源落后时不会用单侧上界删除 StarFX 的最新正常点。
    """
    monotonic_points, rejected = _clean_monotonic_points(starfx_points)
    reference, _ = _clean_monotonic_points(trusted_reference)
    reference_times = [point["time"] for point in reference]

    kept = []
    envelope_rejected = []
    for point in monotonic_points:
        if not reference:
            kept.append(point)
            continue

        position = bisect_right(reference_times, point["time"])
        prev_point = reference[position - 1] if position > 0 else None
        next_point = reference[position] if position < len(reference) else None

        def tolerance(ep: int) -> float:
            return max(float(absolute_tolerance), abs(float(ep)) * float(relative_tolerance))

        # 与可信源恰好同时间戳时应做双侧比较；bisect_right 会把该点放在
        # prev_point，因此不能只把它当作累计值下界。
        if prev_point is not None and prev_point["time"] == point["time"]:
            too_low = point["ep"] < prev_point["ep"] - tolerance(prev_point["ep"])
            too_high = point["ep"] > prev_point["ep"] + tolerance(prev_point["ep"])
        else:
            too_low = (
                prev_point is not None
                and point["ep"] < prev_point["ep"] - tolerance(prev_point["ep"])
            )
            too_high = (
                next_point is not None
                and point["ep"] > next_point["ep"] + tolerance(next_point["ep"])
            )
        if too_low or too_high:
            envelope_rejected.append({**point, "reason": "trusted_source_envelope"})
        else:
            kept.append(point)

    all_rejected = rejected + envelope_rejected
    return kept, {
        "input": len(starfx_points),
        "kept": len(kept),
        "removed": len(all_rejected),
        "rejected": sorted(all_rejected, key=lambda point: point["time"]),
    }


def _choose_time_conflict(candidates: Sequence[Tuple[str, dict]]) -> dict:
    """同一时间戳冲突时优先多数值；平票时按可信源优先级选择。"""
    counts = Counter(point["ep"] for _, point in candidates)
    max_count = max(counts.values())
    majority_values = {ep for ep, count in counts.items() if count == max_count}
    eligible = [(source, point) for source, point in candidates if point["ep"] in majority_values]
    source, point = min(
        eligible,
        key=lambda item: (SOURCE_PRIORITY.get(item[0], 99), item[1]["ep"]),
    )
    return dict(point)


def merge_cutoff_sources(source_points: Dict[str, Sequence[dict]]) -> Tuple[List[dict], dict]:
    """对所有非冲突时间点取并集，并保证最终累计序列非下降。"""
    by_time: Dict[int, List[Tuple[str, dict]]] = defaultdict(list)
    total_points = 0
    for source, points in source_points.items():
        for point in points:
            total_points += 1
            by_time[int(point["time"])].append((source, dict(point)))

    time_conflicts = 0
    merged = []
    for timestamp in sorted(by_time):
        candidates = by_time[timestamp]
        if len({point["ep"] for _, point in candidates}) > 1:
            time_conflicts += 1
        merged.append(_choose_time_conflict(candidates))

    cleaned, rejected = _clean_monotonic_points(merged)
    return cleaned, {
        "input_points": total_points,
        "unique_timestamps": len(by_time),
        "same_time_conflicts": time_conflicts,
        "removed_for_monotonicity": len(rejected),
        "merged": len(cleaned),
    }


def _default_fetch_json(url: str):
    from utils.http_utils import get_json_with_retry

    return get_json_with_retry(url, max_retries=3)


def fetch_and_merge_cutoffs(
    server: int,
    event_id: int,
    tier: int,
    *,
    start_at: Optional[int] = None,
    end_at: Optional[int] = None,
    fetch_json: Optional[Callable[[str], object]] = None,
) -> Tuple[List[dict], dict]:
    """并行获取所有可用数据源，清洗 StarFX 后返回三源并集。"""
    fetch_json = fetch_json or _default_fetch_json
    urls = cutoff_source_urls(server, event_id, tier)
    payloads: Dict[str, object] = {}
    errors: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=len(urls)) as executor:
        futures = {executor.submit(fetch_json, url): source for source, url in urls.items()}
        for future in as_completed(futures):
            source = futures[future]
            try:
                payloads[source] = future.result()
            except Exception as exc:  # 单源失败不能阻断其他源
                errors[source] = f"{type(exc).__name__}: {exc}"

    normalized: Dict[str, List[dict]] = {}
    source_report = {}
    terminal_drop_report = {}
    successful_sources = []
    for source in urls:
        if source in errors:
            source_report[source] = {"status": "error", "error": errors[source]}
            continue
        cutoffs = _extract_cutoffs(payloads.get(source))
        if cutoffs is None:
            source_report[source] = {"status": "invalid_response"}
            continue
        points, normalization_report = normalize_cutoff_points(
            cutoffs, start_at=start_at, end_at=end_at
        )
        points, terminal_report = filter_terminal_drop(points, end_at)
        normalized[source] = points
        terminal_drop_report[source] = terminal_report
        usable = bool(points) or not cutoffs
        if usable:
            successful_sources.append(source)
        source_report[source] = {
            "status": "ok" if usable else "filtered_empty",
            **normalization_report,
            "terminal_drop_removed": terminal_report["removed"],
        }

    trusted = {
        source: points
        for source, points in normalized.items()
        if source in ("bestdori", "hhwx")
    }
    trusted_reference, _ = merge_cutoff_sources(trusted)

    starfx_report = {"input": 0, "kept": 0, "removed": 0, "rejected": []}
    if "starfx" in normalized:
        normalized["starfx"], starfx_report = filter_starfx_outliers(
            normalized["starfx"], trusted_reference
        )
        if not normalized["starfx"] and source_report["starfx"]["received"] > 0:
            source_report["starfx"]["status"] = "filtered_empty"
            if "starfx" in successful_sources:
                successful_sources.remove("starfx")

    merged, merge_report = merge_cutoff_sources(normalized)
    report = {
        "quality_version": QUALITY_VERSION,
        "cutoff_sha256": cutoff_digest(merged),
        "result_status": ("ok" if merged else "structural_zero" if successful_sources else "unknown"),
        "server": int(server),
        "event_id": int(event_id),
        "tier": int(tier),
        "successful_sources": successful_sources,
        "sources": source_report,
        "terminal_drop_filter": terminal_drop_report,
        "starfx_filter": starfx_report,
        "merge": merge_report,
    }
    return merged, report
