# train.py
import os, argparse
from datetime import datetime
import tensorflow as tf
import keras
from config import Config
from data.dataset import load_preprocessed, make_batch_dataset_from_buckets
from model.baseline import SimpleMLPBaseline
from model.seasonal_mlp import SeasonalMLP
from model.cdf_shape import CDFShapeMLP
from model.moe_shape import MoeShapeModel
from utils.tf_ops import set_global_seed
from utils.plotting import plot_and_save_training_curves
from evaluation.visualization import visualize_val_sequences_after_training
import shutil
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="moe",
                    help="模型名称，用于生成固定保存路径")
    args = parser.parse_args()

    config = Config()
    if args.epochs is not None: config.epochs = args.epochs
    if args.batch_size is not None: config.batch_size = args.batch_size
    if args.lr is not None: config.lr = args.lr
    if args.hidden is not None: config.hidden = args.hidden

    set_global_seed(config.seed)

    # 输出目录
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(config.out_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
    shutil.copy(config_file, out_dir)

    # 加载数据
    norm, train_buckets = load_preprocessed(config.pre_h5, "train")
    _, val_buckets = load_preprocessed(config.pre_h5, "val")

    train_ds, train_steps = make_batch_dataset_from_buckets(
        train_buckets, batch_size=config.batch_size, training=True, seed=config.seed, repeat=True
    )
    val_ds, val_steps = make_batch_dataset_from_buckets(
        val_buckets, batch_size=config.batch_size, training=False, seed=config.seed + 999, repeat=False
    )

    print(f"Train steps per epoch: {train_steps}, Val steps: {val_steps}")

    # 构建模型
    model = MoeShapeModel(
        y_mean=norm["y_mean"], y_std=norm["y_std"],
        ctx_min=norm["ctx_min"], ctx_max=norm["ctx_max"],
        hidden=config.hidden, dropout=config.dropout,
        n_experts=config.n_experts,
        gating_topk=config.gating_topk,
        gating_temperature=config.gating_temp,
        gating_noise_std=config.gating_noise_std,
        bias_update_speed=config.bias_speed,
        season_resid_l2=config.season_resid_l2,
        cdf_mix_power=config.cdf_mix_power,
        gate_dropout=config.gate_dropout,
        name="MoeShapeModel"
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.lr))

    # 可选：先用一个 batch 构建模型
    x0, y0, w0 = next(iter(train_ds.take(1)))
    _ = model(x0, training=False)

    # 训练
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_sum_loss", patience=3, restore_best_weights=True,
            mode="min", min_delta=1e-4, verbose=1
        )
    ]

    history = model.fit(
        train_ds,
        epochs=config.epochs,
        steps_per_epoch=train_steps,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=callbacks,
    )

    # 保存
    plot_and_save_training_curves(history, out_dir)
    model_name = args.model_name if args.model_name is not None else "model"
    model.save(os.path.join(out_dir, f"{model_name}.keras"))
    print(f"Model and plots saved to {out_dir}")
    fixed_model_dir = "saved_models"
    os.makedirs(fixed_model_dir, exist_ok=True)
    fixed_model_path = os.path.join(fixed_model_dir, f"{model_name}.keras")
    model.save(fixed_model_path)
    print(f"Model also saved to {fixed_model_path}")
    # 训练完成后进行可视化

    visualize_val_sequences_after_training(
        model=model,
        raw_h5=config.raw_h5,          # 使用 config 中的 data.hdf5 路径
        out_dir=out_dir,
        n_seq=1000,
        split="val",
        t_min=config.t_min,            # 与预处理一致的最小历史长度
        max_slices=1024,
        chunk_size=256,
        seed=config.seed,
    )

    print(f"训练和可视化全部完成，输出目录: {out_dir}")

if __name__ == "__main__":
    main()