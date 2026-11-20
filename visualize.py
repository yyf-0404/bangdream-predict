# visualize.py
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端

from config import CURVE_DATA
from model.conditional_curve import DEFAULT_MODEL_PATH

def main():
    parser = argparse.ArgumentParser(description="生成验证集切片可视化")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH), help="默认使用随仓库发布的条件曲线权重")
    parser.add_argument("--raw_h5", type=str, help="与模型对应的逐档或联合数据 HDF5")
    parser.add_argument("--out_dir", type=str, default="outputs/viz_test", help="可视化输出目录")
    parser.add_argument("--n_seq", type=int, default=1000, help="可视化的序列数量")
    parser.add_argument("--t_min", type=int, default=48, help="最小历史长度")
    parser.add_argument("--max_slices", type=int, default=1024, help="每条序列的最大切片数")
    parser.add_argument("--chunk_size", type=int, default=256, help="分块推理大小")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--raw_dir", default="datasets/raw", help="条件曲线回放的带时间戳档线目录")
    parser.add_argument("--final_labels", default=str(Path(__file__).parent / 'data/splits/cn_verified_finals_through_328.json'),
                        help="核验结算分 JSON，行字段为 event_id、tier、final")
    parser.add_argument("--workers", type=int, default=4, help="条件曲线按活动并行的进程数")
    args = parser.parse_args()

    if Path(args.model_path).suffix == ".npz":
        from evaluation.conditional_visualization import visualize_conditional_validation
        visualize_conditional_validation(args.model_path, args.raw_h5 or CURVE_DATA,
                                         args.raw_dir, args.out_dir, final_labels=args.final_labels,
                                         n_seq=args.n_seq, t_min=args.t_min, max_slices=args.max_slices,
                                         seed=args.seed, workers=args.workers)
        return

    # 导入所有自定义层以确保模型可以加载
    from evaluation.visualization import visualize_val_sequences_after_training
    import tensorflow as tf
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
        raw_h5=args.raw_h5 or "datasets/data.hdf5",
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
