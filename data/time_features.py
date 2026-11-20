"""Explicit business time zones; no dependency on the host TZ setting."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TIME_FEATURE_VERSION = "server-timezone-v1"
SERVER_TIMEZONES = {
    0: "Asia/Tokyo", 1: "UTC", 2: "Asia/Taipei",
    3: "Asia/Shanghai", 4: "Asia/Seoul",
}


def server_datetime(timestamp_ms, server=3):
    if server not in SERVER_TIMEZONES:
        raise ValueError(f"Unsupported server timezone: {server}")
    return datetime.fromtimestamp(int(timestamp_ms) / 1000, ZoneInfo(SERVER_TIMEZONES[server]))


def time_features(timestamps_ms, server=3, holidays=None):
    """Raw categories: hour 0..47, weekday 0..6, holiday state 0..3."""
    import numpy as np
    if holidays is None:
        from .constants import get_cn_holidays
        holidays = get_cn_holidays() if server == 3 else set()
    rows = []
    for timestamp in timestamps_ms:
        local = server_datetime(timestamp, server)
        today = local.date()
        is_hol = today in holidays
        next_hol = today + timedelta(days=1) in holidays
        day_type = (2 if next_hol else 3) if is_hol else (1 if next_hol else 0)
        rows.append([local.hour * 2 + local.minute // 30, local.weekday(), day_type])
    return np.asarray(rows, dtype=np.int32).reshape(-1, 3)
