# utils/plotting.py
import os, json
import matplotlib.pyplot as plt

def plot_and_save_training_curves(history, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    hist = history.history or {}
    # 保存 JSON
    out_json = {k: [float(x) for x in v] for k, v in hist.items()}
    with open(os.path.join(out_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    # 绘制总损失
    plt.figure()
    if "loss" in hist:
        plt.plot(hist["loss"], label="train_loss")
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Training Curves")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()