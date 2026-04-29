# visualize.py
import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端


from data.constants import inv_eventType, tierListOfServer, bandNames, serverNames
from evaluation.visualization import visualize_val_sequences_after_training

def main():
    parser = argparse.ArgumentParser(description="生成验证集切片可视化")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的 .keras 模型路径")
    parser.add_argument("--raw_h5", type=str, default="datasets/data.hdf5", help="原始数据 HDF5 路径")
    parser.add_argument("--out_dir", type=str, default="outputs/viz_test", help="可视化输出目录")
    parser.add_argument("--n_seq", type=int, default=1000, help="可视化的序列数量")
    parser.add_argument("--t_min", type=int, default=48, help="最小历史长度")
    parser.add_argument("--max_slices", type=int, default=1024, help="每条序列的最大切片数")
    parser.add_argument("--chunk_size", type=int, default=256, help="分块推理大小")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    # 导入所有自定义层以确保模型可以加载
    from model.moe_shape import MoeShapeModel
    from model.layers.moe import ShapeMoE
    from model.layers.seasonal import SeasonHourSegNet, WeekHolCorrectionNet
    from model.layers.normalization import Log1pStandardize, MinMaxNormalizeClip

    print("加载模型...")
    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={
            'MoeShapeModel': MoeShapeModel,
            'ShapeMoE': ShapeMoE,
            'SeasonHourSegNet': SeasonHourSegNet,
            'WeekHolCorrectionNet': WeekHolCorrectionNet,
            'Log1pStandardize': Log1pStandardize,
            'MinMaxNormalizeClip': MinMaxNormalizeClip,
        }
    )

    print("开始生成可视化...")
    visualize_val_sequences_after_training(
        model=model,
        raw_h5=args.raw_h5,
        out_dir=args.out_dir,
        n_seq=args.n_seq,
        split="val",
        t_min=args.t_min,
        max_slices=args.max_slices,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )
    print(f"可视化完成，图片保存在 {args.out_dir}/val_slice_viz/")

if __name__ == "__main__":
    main()