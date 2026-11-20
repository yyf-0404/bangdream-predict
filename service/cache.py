# service/cache.py
import os
import json
import threading
from typing import Dict, List, Optional
from utils.http_utils import save_json
from data.rank_order import CUTOFF_SEQUENCE_VERSION
from data.time_features import TIME_FEATURE_VERSION

class PredictionCache:
    def __init__(self, cache_dir: Optional[str] = None):
        self._lock = threading.RLock()
        # Preserve legacy caches on disk, but never reuse uncorrected PCHIP
        # predictions after changing cumulative-grid semantics.
        self.cache_dir = cache_dir if cache_dir is not None else os.path.join(
            "service/cache_data", TIME_FEATURE_VERSION, CUTOFF_SEQUENCE_VERSION,
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        # 内存缓存（用于快速访问）
        self._history: Dict[str, List[dict]] = {}
        self._t0_sets: Dict[str, List[int]] = {}
        self._latest_slice: Dict[str, dict] = {}
        self._latest_t0: Dict[str, int] = {}

        # 启动时从文件加载所有已有缓存
        self._load_all()

    def _make_key(self, event_id: int, server: int, tier: int) -> str:
        return f"{event_id}_{server}_{tier}"

    # ----- 文件路径 -----
    def _history_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, "history", *(f"{key}.json").split("_"))

    def _latest_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, "latest", *(f"{key}.json").split("_"))

    # ----- 加载与保存 -----
    def _load_all(self):
        """启动时扫描 history 和 latest 子目录，加载所有缓存到内存。"""
        history_dir = os.path.join(self.cache_dir, "history")
        latest_dir = os.path.join(self.cache_dir, "latest")

        with self._lock:
            # 加载历史缓存
            if os.path.isdir(history_dir):
                for dirpath, _, filenames in os.walk(history_dir):
                    for fname in filenames:
                        if fname.endswith(".json"):
                            full_path = os.path.join(dirpath, fname)
                            # 从相对路径重建 key: history/eventId/server/tier.json -> eventId_server_tier
                            rel = os.path.relpath(full_path, history_dir)
                            key = rel.replace(os.sep, "_").removesuffix(".json")
                            try:
                                with open(full_path, "r") as f:
                                    data = json.load(f)
                                self._history[key] = data.get("history", [])
                                self._t0_sets[key] = data.get("t0_set", [])
                            except Exception:
                                print(f"Failed to load history cache: {full_path}")

            # 加载最新切片缓存
            if os.path.isdir(latest_dir):
                for dirpath, _, filenames in os.walk(latest_dir):
                    for fname in filenames:
                        if fname.endswith(".json"):
                            full_path = os.path.join(dirpath, fname)
                            rel = os.path.relpath(full_path, latest_dir)
                            key = rel.replace(os.sep, "_").removesuffix(".json")
                            try:
                                with open(full_path, "r") as f:
                                    data = json.load(f)
                                self._latest_slice[key] = data.get("slice", {})
                                self._latest_t0[key] = data.get("t0", 0)
                            except Exception:
                                print(f"Failed to load latest cache: {full_path}")

    def _save_history(self, key: str):
        path = self._history_path(key)

        save_json({
            "history": self._history.get(key, []),
            "t0_set": self._t0_sets.get(key, []),
        }, path)

    def _save_latest(self, key: str):
        path = self._latest_path(key)
        save_json({
            "slice": self._latest_slice.get(key, {}),
            "t0": self._latest_t0.get(key, 0),
        }, path)

    # ----- 公共接口（线程安全）-----
    def get_history(self, event_id: int, server: int, tier: int) -> Optional[List[dict]]:
        with self._lock:
            return self._history.get(self._make_key(event_id, server, tier))

    def get_t0_set(self, event_id: int, server: int, tier: int) -> List[int]:
        key = self._make_key(event_id, server, tier)
        with self._lock:
            return list(self._t0_sets.get(key, []))

    def get_latest_slice(self, event_id: int, server: int, tier: int) -> Optional[dict]:
        key = self._make_key(event_id, server, tier)
        with self._lock:
            return self._latest_slice.get(key)

    def get_latest_t0(self, event_id: int, server: int, tier: int) -> Optional[int]:
        key = self._make_key(event_id, server, tier)
        with self._lock:
            return self._latest_t0.get(key)

    def update_history(
        self,
        event_id: int, server: int, tier: int,
        new_entries: List[dict],
        t0_list: List[int],
    ):
        key = self._make_key(event_id, server, tier)
        with self._lock:
            existing = self._history.get(key, [])
            merged = {e["t0"]: e for e in existing}
            for entry, t0 in zip(new_entries, t0_list):
                merged[t0] = entry
            sorted_entries = [merged[t] for t in sorted(merged.keys())]
            self._history[key] = sorted_entries
            self._t0_sets[key] = sorted(merged.keys())
            self._save_history(key)

    def update_latest_slice(
        self,
        event_id: int, server: int, tier: int,
        latest_slice: dict,
        t0: int,
    ):
        key = self._make_key(event_id, server, tier)
        with self._lock:
            self._latest_slice[key] = latest_slice
            self._latest_t0[key] = t0
            self._save_latest(key)

    def clear(self):
        with self._lock:
            self._history.clear()
            self._t0_sets.clear()
            self._latest_slice.clear()
            self._latest_t0.clear()
            import shutil
            for sub in ["history", "latest"]:
                path = os.path.join(self.cache_dir, sub)
                if os.path.exists(path):
                    shutil.rmtree(path)
