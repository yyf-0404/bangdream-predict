# service/updater.py
import logging
import threading
from typing import List, Optional
from datetime import datetime, timezone
from data.raw import (
    generate_sequences,
    _cache_metadata,
    get_valid_event_ids,
    get_active_events,
    get_start_at,
    get_end_at,
    get_latest_cutoff_time,
    download_cutoffs
)
from data.constants import oldEventTierListOfServer, step_length, tier_list_for_event
from data.constants import is_supported_tier
from .cache import PredictionCache
from .predictor import Predictor

logger = logging.getLogger(__name__)


class DataUpdater:
    def __init__(
        self,
        server_list: List[int] = None,
        tier_list: List[int] = None,
        update_interval: int = 3600,
        t_min: int = 48,
        batch_size: int = 256,
        min_event_id: int = 226,
    ):
        self.server_list = server_list or [3]
        self.tier_list = tier_list
        self.update_interval = update_interval
        self.t_min = t_min
        self.batch_size = batch_size
        self.min_event_id = min_event_id

        self.cache = PredictionCache()
        self.predictor: Optional[Predictor] = None
        self._stop_event = threading.Event()

    def set_predictor(self, predictor: Predictor):
        self.predictor = predictor

    def get_predictions(self, event_id, server, tier):
        latest = self.ensure_latest_slice(event_id, server, tier)
        history = self.cache.get_history(event_id, server, tier)
        if history is None:
            return None
        return dict(event_id=event_id, server=server, tier=tier, history=history, latest_slice=latest)

    # -----------------------------------------------------------------
    # 按需生成序列（不保存到内存）
    # -----------------------------------------------------------------
    def _generate_sequences_for_events(self, event_ids: List[int]) -> list:
        if not event_ids:
            return []
        logger.debug(f"Generating sequences for events {event_ids[0]} ~ {event_ids[-1]}")
        return generate_sequences(
            event_ids=event_ids,
            server_list=self.server_list,
            tier_list=self.tier_list,
            split_threshold=10000,
        )

    # -----------------------------------------------------------------
    # 填充历史切点缓存（无 latest_slice）
    # -----------------------------------------------------------------
    def _fill_history_cache(self, seqs: list):
        if self.predictor is None:
            raise RuntimeError("Predictor not set")
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        for seq in seqs:
            try:
                event_id = int(seq["context_cat"][4])
                server = int(seq["context_cat"][3])
                tier_idx = int(seq["context_cat"][2])
                tier_val = seq.get("tier")
                tier_val = int(tier_val) if tier_val is not None else self._tier_index_to_value(tier_idx)

                T = len(seq["y"])
                if T <= self.t_min:
                    continue

                start_at = get_start_at(server=server, eventId=event_id)
                end_at = get_end_at(server=server, eventId=event_id)   # 确保已定义此方法

                # 计算当前时间对应的步数（若活动已结束，则取 T）
                if now_ms < end_at:
                    latest_time = get_latest_cutoff_time(event_id, server, tier_val)
                    if latest_time is not None:
                        current_step = max(0, (latest_time - start_at) // step_length)
                        current_t0 = int(min(current_step, T))
                    else:
                        current_t0 = self.t_min   # 无数据时不预先填充任何历史切点
                else:
                    current_t0 = T

                # 填充上限：已结束的活动用 T，进行中的用 current_t0
                max_t0 = current_t0
                if max_t0 <= self.t_min:
                    continue

                all_t0 = list(range(self.t_min, max_t0))
                cached_t0 = set(self.cache.get_t0_set(event_id, server, tier_val))
                missing_t0 = [t for t in all_t0 if t not in cached_t0]

                if missing_t0:
                    logger.info(
                        f"Event {event_id}/tier {tier_val}: filling {len(missing_t0)} missing slices"
                    )
                    raw_entries = self.predictor.predict_slices(
                        seq, missing_t0, batch_size=self.batch_size
                    )
                    new_entries = []
                    for entry, t0 in zip(raw_entries, missing_t0):
                        ep = int(entry.get("total_sum_hat", 0))
                        time_ms = int(start_at + t0 * step_length)
                        new_entries.append({
                            "t0": t0,
                            "ep": ep,
                            "time": time_ms,
                        })
                    self.cache.update_history(
                        event_id, server, tier_val,
                        new_entries=new_entries,
                        t0_list=missing_t0,
                    )
            except Exception as e:
                logger.error(f"Error filling history for event {event_id}: {e}", exc_info=True)
    
    # -----------------------------------------------------------------
    # 公共方法：确保 latest_slice 为最新，并返回
    # -----------------------------------------------------------------
    def ensure_latest_slice(self, event_id: int, server: int, tier: int,
                            seq: Optional[dict] = None) -> Optional[dict]:
        if self.predictor is None:
            logger.error("Predictor not set")
            return None
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        try:
            start_at = get_start_at(server, event_id)
            end_at = get_end_at(server, event_id)
            if start_at is None or end_at is None:
                logger.warning(
                    "Missing event timing for event %s, server %s; skip prediction.",
                    event_id, server
                )
                return None
            if not (start_at <= now_ms < end_at):
                # 活动未开始或已结束，不处理
                return None
            if not is_supported_tier(server, event_id, tier):
                logger.warning(
                    "Unsupported tier %s for event %s, server %s; skip prediction.",
                    tier, event_id, server
                )
                return None

            latest_time = get_latest_cutoff_time(event_id, server, tier)
            if latest_time is None or latest_time < now_ms - step_length:
                logger.info(
                    "Cutoff missing or stale for event %s, server %s, tier %s; downloading.",
                    event_id, server, tier
                )
                download_cutoffs(
                    event_list=[event_id],
                    server_list=[server],
                    tier_list=[tier],
                    force=True,
                )
                # The supplied event grid predates this download. Regenerate
                # it (including peer ranks) before using a newer slice limit.
                seq = None
                latest_time = get_latest_cutoff_time(event_id, server, tier)
                if latest_time is None:
                    return None
            elapsed = latest_time - start_at
            current_t0 = max(0, elapsed // step_length)
            # 确保不超过理论最大步数
            end_at = get_end_at(server, event_id)
            T = max(0, (end_at - start_at) // step_length)
            current_t0 = min(current_t0, T)

            if current_t0 < self.t_min:
                return None

            cached_history_t0 = set(self.cache.get_t0_set(event_id, server, tier))
            cached_latest_t0 = self.cache.get_latest_t0(event_id, server, tier)
            if cached_latest_t0 == current_t0 and current_t0 in cached_history_t0:
                # 缓存已完整，无需推理
                return self.cache.get_latest_slice(event_id, server, tier)
            # 未传入 seq 时，按需生成序列
            if seq is None:
                seq = self._generate_single_sequence(event_id, server, tier)
                if seq is None:
                    return None
            curve_info = self.predictor.predict_current_curve(seq, current_t0)
            
            full_prefix = curve_info.get("prefix_curve")  # 长度 T
            t0 = curve_info["t0"]
            future_len = curve_info["future_len"]
            if full_prefix is not None:
                # 计算未来各步的累积值（从索引 t0 开始）
                future_prefix = full_prefix[t0:]  # 长度 future_len
                # 计算时间戳：start_at + (t0 + i + 1) * step_length
                timestamps = [int(start_at + (t0 + i + 1) * step_length) for i in range(future_len)]
                new_prefix = [{"time": ts, "ep": int(cum)} for ts, cum in zip(timestamps, future_prefix)]
                curve_info["prefix_curve"] = new_prefix
            
            # 更新历史缓存
            if current_t0 not in cached_history_t0:
                history_entry = {
                    "t0": current_t0,
                    "ep": int(curve_info["total_sum_hat"]),
                    "time": int(start_at + current_t0 * step_length)   # 新增
                }
                self.cache.update_history(event_id, server, tier,
                                        new_entries=[history_entry],
                                        t0_list=[current_t0])

            # 更新最新切片缓存
            if cached_latest_t0 != current_t0:
                self.cache.update_latest_slice(event_id, server, tier, curve_info, current_t0)
                logger.info(f"Updated latest slice for event {event_id}, tier {tier}, t0={current_t0}")

            return curve_info
        except Exception as e:
            logger.error(f"Error in ensure_latest_slice: {e}", exc_info=True)
            return None

    def _generate_single_sequence(self, event_id: int, server: int, tier: int):
        """生成指定活动/服务器/档位的序列（通常只有一条）。"""
        try:
            seqs = generate_sequences(
                event_ids=[event_id],
                server_list=[server],
                tier_list=[tier],
                split_threshold=10000,
            )
            if seqs:
                return seqs[0]
        except Exception as e:
            logger.error(f"Failed to generate sequence for event {event_id}, tier {tier}: {e}")
        return None
    # -----------------------------------------------------------------
    # 启动初始化：处理所有有效活动，填充历史+最新切片
    # -----------------------------------------------------------------
    def initialize_cache(self):
        """仅对预测不完整的活动及档位生成序列，填充缓存。"""
        _cache_metadata(True)
        server = 3
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        valid_ids = get_valid_event_ids(
            server=server,
            min_event_id=self.min_event_id,
            current_time_ms=now_ms,
        )
        if not valid_ids:
            logger.info("No valid events to process at startup.")
            return

        # 计算每个活动的理论总步数
        event_length = {}
        for ev_id in valid_ids:
            start = get_start_at(server, ev_id)
            end = get_end_at(server, ev_id)
            if not start or not end:
                continue
            T = max(0, (end - start) // step_length)
            if T > self.t_min:
                event_length[ev_id] = T

        # 收集预测不完整的活动
        needed_events = set()
        for ev_id, T in event_length.items():
            all_t0 = set(range(self.t_min, T))
            for tier in self._tier_list_for_event(server, ev_id):
                if not is_supported_tier(server, ev_id, tier):
                    continue
                cached_t0 = set(self.cache.get_t0_set(ev_id, server, tier))
                if all_t0 - cached_t0:
                    needed_events.add(ev_id)
                    break

        if not needed_events:
            logger.info("All valid events already have complete prediction cache.")
            return

        logger.info(f"Found {len(needed_events)} events with incomplete cache: "
                    f"{min(needed_events)} ~ {max(needed_events)}")
        download_cutoffs(event_list=sorted(needed_events), force=True)
        seqs = generate_sequences(
            event_ids=sorted(needed_events),
            server_list=self.server_list,
            tier_list=self.tier_list,
            split_threshold=10000,
        )

        # 批量填充缺失的历史切点（包含 time 字段）
        self._fill_history_cache(seqs)
        active_ids = set(get_active_events(3, current_time_ms=now_ms))
        # 生成初始的最新切片（对进行中的活动）
        for seq in seqs:
            event_id = int(seq["context_cat"][4])
            if event_id not in active_ids:
                continue
            server = int(seq["context_cat"][3])
            tier_idx = int(seq["context_cat"][2])
            tier_val = seq.get("tier")
            tier_val = int(tier_val) if tier_val is not None else self._tier_index_to_value(tier_idx)
            self.ensure_latest_slice(event_id, server, tier_val, seq=seq)

        logger.info("Initial cache fill completed.")
    # -----------------------------------------------------------------
    # 定期更新：只针对活跃活动调用 ensure_latest_slice
    # -----------------------------------------------------------------
    def periodic_update(self):
        if self.predictor is None:
            return
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        active_events = get_active_events(3, current_time_ms=now_ms)
        if not active_events:
            logger.debug("No active events for periodic update.")
            return
        logger.info(f"Active events: {active_events}")

        # 按活动规则版本选择档位：旧活动无 1500，新活动包含 1500。
        for event_id in active_events:
            for tier in self._tier_list_for_event(3, event_id):
                if not is_supported_tier(3, event_id, tier):
                    continue
                # 调用公共方法即可，它会自动更新历史+latest
                self.ensure_latest_slice(event_id, 3, tier)

    # -----------------------------------------------------------------
    # 后台线程
    # -----------------------------------------------------------------
    def _run_periodic_update(self):
        while not self._stop_event.is_set():
            try:
                _cache_metadata(True)  # 定期刷新元数据缓存
                self.periodic_update()
            except Exception as e:
                logger.error(f"Periodic task failed: {e}", exc_info=True)
            self._stop_event.wait(self.update_interval)

    def start(self):
        self.initialize_cache()
        thread = threading.Thread(target=self._run_periodic_update, daemon=True)
        thread.start()

    def stop(self):
        self._stop_event.set()

    # -----------------------------------------------------------------
    # 辅助
    # -----------------------------------------------------------------
    def _tier_list_for_event(self, server: int, event_id: int):
        if self.tier_list is not None:
            return self.tier_list
        return tier_list_for_event(server, event_id)

    def _tier_index_to_value(self, tier_idx: int) -> int:
        tier_list = oldEventTierListOfServer.get(3, [])
        if 0 <= tier_idx < len(tier_list):
            return tier_list[tier_idx]
        return tier_idx
