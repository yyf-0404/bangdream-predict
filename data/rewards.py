"""Content-based ranking rewards. IDs/stage names are audit-only, never tokens."""
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np

from .time_features import server_datetime

DEFAULT_REWARD_TABLE = Path(__file__).resolve().parent / "rewards" / "cn_rank_rewards_v1.json"
REWARD_FEATURE_NAMES = ("stars_per_1000", "voice", "dynamic", "flash",
                        "distance_stars", "distance_voice", "distance_dynamic", "distance_flash")
REWARD_FEATURE_VERSION = "four-rewards-log-distance-v1"


class RewardTable:
    def __init__(self, path=DEFAULT_REWARD_TABLE, *, document=None):
        if document is None:
            with open(path, encoding="utf-8") as handle:
                self.document = json.load(handle)
        else:
            # Saved V2 models carry their exact table, independently of local updates.
            self.document = json.loads(json.dumps(document, allow_nan=False))
        table = self.document
        if table.get("server") != 3 or not table.get("version"):
            raise ValueError("Expected a versioned CN reward table")
        if (table.get("min_rank"), table.get("max_rank")) != (1, 100000):
            raise ValueError("Reward table must explicitly cover ranks 1..100000")
        if type(table.get("first_supported_event")) is not int or table["first_supported_event"] < 1:
            raise ValueError("Invalid reward coverage start")
        self.version = table["version"]
        self.sha256 = hashlib.sha256(json.dumps(table, ensure_ascii=False, sort_keys=True,
                                               separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.stages = table["stages"]
        if not self.stages or self.stages[0].get("effective_from") is not None:
            raise ValueError("Reward table needs a baseline stage")
        previous_date, names = date.min, set()
        for i, stage in enumerate(self.stages):
            current_date = date.fromisoformat(stage["effective_from"]) if i else date.min
            if (i and current_date <= previous_date) or stage["name"] in names:
                raise ValueError("Reward stages must have unique names and increasing dates")
            previous_date = current_date
            names.add(stage["name"])
            last_rank, last_amount = 0, float("inf")
            for rank, amount in stage["stars"]:
                if (type(rank) is not int or not last_rank < rank <= 100000
                        or type(amount) is not int or not 0 <= amount <= last_amount):
                    raise ValueError("Invalid star reward intervals")
                last_rank, last_amount = rank, amount
            if last_rank != 100000:
                raise ValueError("Incomplete star reward coverage")
            for key in ("voice_top", "dynamic_top", "flash_top"):
                value = stage[key]
                if value is not None and (type(value) is not int or not 1 <= value <= 100000):
                    raise ValueError(f"Invalid reward threshold: {key}")

    def encode(self, server, event_id, start_at, ranks):
        if server != 3 or event_id < self.document["first_supported_event"]:
            raise ValueError("Event is outside confirmed CN reward coverage")
        ranks = np.asarray(ranks)
        if (ranks.ndim != 1 or not len(ranks) or not np.issubdtype(ranks.dtype, np.integer)
                or (ranks < 1).any() or (ranks > 100000).any()):
            raise ValueError("Reward query ranks must be integers in 1..100000")
        local_date = server_datetime(start_at, server).date()
        stage = self.stages[0]
        for candidate in self.stages[1:]:
            if local_date >= date.fromisoformat(candidate["effective_from"]):
                stage = candidate
        values = np.zeros((len(ranks), 4), dtype=np.float32)
        bounds = np.array([row[0] for row in stage["stars"]])
        amounts = np.array([row[1] for row in stage["stars"]])
        values[:, 0] = amounts[np.searchsorted(bounds, ranks)] / 1000.
        thresholds = [stage[key] for key in ("voice_top", "dynamic_top", "flash_top")]
        for column, top in enumerate(thresholds, 1):
            if top is not None:
                values[:, column] = ranks <= top
        # No artificial boundary at max_rank: rewards beyond our coverage are unknown.
        star_boundaries = [bounds[i] for i in range(len(bounds) - 1) if amounts[i] != amounts[i + 1]]
        boundaries = [star_boundaries] + [[top] if top is not None and top < 100000 else []
                                          for top in thresholds]
        distances = np.zeros_like(values)
        distance_mask = np.zeros_like(values, dtype=bool)
        for column, edges in enumerate(boundaries):
            if edges:
                all_distances = np.log1p(ranks[:, None]) - np.log1p(np.asarray(edges)[None, :])
                # Sorted boundaries + argmin give the smaller boundary on a tie.
                nearest = np.abs(all_distances).argmin(axis=1)
                distances[:, column] = all_distances[np.arange(len(ranks)), nearest]
                distance_mask[:, column] = True
        return {
            "reward_features": np.concatenate([values, distances], axis=1),
            "reward_feature_mask": np.concatenate([np.ones_like(values, dtype=bool), distance_mask], axis=1),
            "reward_stage": stage["name"],
            "reward_schedule_version": self.version,
            "reward_schedule_sha256": self.sha256,
        }
