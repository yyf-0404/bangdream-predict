# config.py
from dataclasses import dataclass
from typing import Optional, List

CURVE_DATA = "datasets/joint_v2_through_328.h5"
CURVE_SPLIT = "data/splits/cn_type_balanced_through_328_v3.json"


@dataclass
class Config:
    # ---------- 数据 ----------
    pre_h5: str = "datasets/preprocessed.h5"       # 预处理后的 HDF5
    raw_h5: str = "datasets/data.hdf5"             # 原始 HDF5（预处理输入）
    raw_dir: str = "datasets/raw"    # 原始缓存数据目录
    split_manifest: str = CURVE_SPLIT
    balance_rule_weights: bool = True
    tier_1500_weight: float = 1.0
    split_threshold: int = 290      # 活动ID ≤ 此值归训练集，其他归验证集

    # 训练用
    batch_size: int = 512
    epochs: int = 10
    lr: float = 1e-4
    seed: int = 1

    # 模型
    hidden: int = 128
    dropout: float = 0.2
    # CDF 混合参数
    cdf_mix_power: float = 4   # CDF 混合
    # MoE 参数
    gate_dropout: float = 0.1   # gate_in 的 dropout 率
    n_experts: int = 6
    gating_topk: int = 4          # 若 >0 且 < n_experts，则只保留 top‑k 个专家
    gating_temp: float = 2.0
    gating_noise_std: float = 0.2   # 训练时噪声
    bias_speed: float = 0.1
    season_resid_l2: float = 1e-5        # 专家季节残差头的 L2 正则化系数

    # 预处理
    t_min: int = 48                       # 最小历史长度
    random_slices_per_seq: Optional[int] = None  # None = 全部切点
    length_mode: str = "hist"             # 按 hist 长度分桶
    weight_mode: str = "sqrt_floor"       # 样本权重模式
    weight_floor: float = 0.2
    bucket_boundaries: List[int] = None   # 动态默认

    # 档位加权
    tier_center: int = 9           # 中心档位索引（对应实际档位 1000）
    tier_alpha: float = 0.5       # 衰减系数
    tier_floor: float = 0.1        # 最低权重，防止远端档位被完全忽略

    # 输出
    out_dir: str = "outputs"

    def __post_init__(self):
        if self.bucket_boundaries is None:
            self.bucket_boundaries = [160, 224, 320, 448, 688]