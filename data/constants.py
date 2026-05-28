# data/constants.py
import os
import holidays

# 中国节假日
cn_holidays = holidays.CountryHoliday('CN')

# 旧活动 tier 列表（只用了 server=3）
oldEventTierListOfServer = {
    3: [20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000,
        10000, 20000, 30000, 50000, 70000, 100000]
}

# 新活动 tier 列表：CN 在活动 310、311 以及大于 313 的活动中新增 1500 档。
newEventTierListOfServer = {
    3: [20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 3000, 4000,
        5000, 10000, 20000, 30000, 50000, 70000, 100000]
}

# 默认 tier 列表保持旧活动规则，用于既有可视化和旧数据索引。
tierListOfServer = oldEventTierListOfServer

MISSING_TIER_IDX = -1

# tier → 旧活动规则下的索引映射。现有模型用这套索引接收新旧活动。
tierMap = {v: i for i, v in enumerate(tierListOfServer[3])}
newEventTierMap = {v: i for i, v in enumerate(newEventTierListOfServer[3])}

def uses_new_event_tiers(server: int, event_id: int) -> bool:
    return server == 3 and (event_id in (310, 311) or event_id > 313)

def tier_list_for_event(server: int, event_id: int):
    if uses_new_event_tiers(server, event_id):
        return newEventTierListOfServer.get(server, [])
    return oldEventTierListOfServer.get(server, [])

def is_supported_tier(server: int, event_id: int, tier: int) -> bool:
    return tier in tier_list_for_event(server, event_id)

def model_tier_index(tier: int) -> int:
    """映射到现有模型可接受的 tier index；未知档位映射到 missing embedding 0。"""
    return tierMap.get(tier, MISSING_TIER_IDX)

# Bestdori API 基础 URL
Bestdoriurl = 'https://bestdori.com'

# 事件类型映射
eventTypeMap = {
    "story": 1,
    "versus": 2,
    "live_try": 3,
    "challenge": 4,
    "mission_live": 5,
    "festival": 6,
    "medley": 7
}

# 时间步长：30分钟（毫秒）
step_length = 30 * 60 * 1000
t_min = 48  # 最小历史长度（单位：切点数）
# 每个角色所属乐队映射（用于 get_band）
bandMap = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 18:6, 21:7, 45:8}

# 反向映射：整数 -> 事件类型字符串 (例如 1 -> 'story')
inv_eventType = {v: k for k, v in eventTypeMap.items()}

# 服务器名称映射（只用到 server 3）
serverNames = {3: "cn"}

# 乐队名称映射（索引 0~8）
bandNames = {
    0: "Mixed",
    1: "ppp",
    2: "ag",
    3: "pp",
    4: "rose",
    5: "hhw",
    6: "ras",
    7: "monica",
    8: "go"
}

# 外生特征维度（用于 Embedding 的输入大小 = max_value + 1，因为 0 保留给 padding）
HOUR_BINS = 48            # 半小时桶数（0~47，加 padding 为 49）
WEEK_BINS = 7             # 星期几（0~6，加 padding 为 8）
HOL_BINS = 4              # 节日四分类（0~3，加 padding 为 5）

# 上下文分类特征的类别数（用于 Embedding 输入维度）
NUM_EVENT_TYPES = 9       # eventType 映射后 + padding
NUM_BANDS = 10            # band 索引 + padding
NUM_TIERS = 21            # 旧活动规则：20 个档位 + padding/missing
NUM_SERVERS = 3 + 2       # server 原值最大为 3，+1 后最大为 4，加 padding 为 5
