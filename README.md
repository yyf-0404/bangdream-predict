# bangdream-predict

BanG Dream! 活动档线预测项目。项目从 Bestdori 获取活动 cutoff 数据，将原始 JSON 转换为 HDF5 序列，训练 TensorFlow/Keras 模型，并通过 FastAPI 提供在线预测服务。

## 功能概览

- 下载并缓存 Bestdori 活动、角色和 cutoff 数据
- 将活动档线转换为 30 分钟粒度的时间序列
- 生成训练用 HDF5 数据集和按历史长度分桶的预处理数据
- 训练多种 Keras 模型，当前默认使用 MoE shape 模型
- 保存训练曲线、验证集切片可视化和 `.keras` 模型
- 通过 FastAPI 返回指定活动、服务器、档位的历史预测和最新预测曲线

## 项目结构

```text
.
├── config.py              # 全局配置：数据路径、训练参数、模型参数
├── download_data.py       # 下载 Bestdori 原始 cutoff 数据
├── generate_raw_h5.py     # 原始 JSON -> datasets/data.hdf5
├── preprocess.py          # datasets/data.hdf5 -> datasets/preprocessed.h5
├── train.py               # 模型训练入口
├── visualize.py           # 加载模型并生成验证集可视化
├── serve.py               # FastAPI 服务启动入口
├── data/                  # 数据下载、序列构造、切片、预处理、tf.data
├── model/                 # Keras 模型和自定义层
├── service/               # 在线预测、后台更新、本地缓存
├── evaluation/            # 可视化评估逻辑
├── utils/                 # HTTP、绘图、TensorFlow 工具
└── saved_models/          # 默认模型目录
```

## 环境准备

建议在虚拟环境中安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

依赖版本见 [requirements.txt](requirements.txt)。如果 TensorFlow 安装失败，请使用与 `tensorflow>=2.20.0,<2.21.0` 兼容的 Python 版本。

## 数据目录

原始数据目录固定由 [config.py](config.py) 中的 `Config.raw_dir` 控制，默认是：

```text
datasets/raw
```

项目不再支持通过命令行或环境变量自定义 raw 数据目录。如需修改，请改 `config.py`。

`.gitignore` 默认忽略以下运行产物：

- `datasets/`
- `outputs/`
- `service/cache_data/`
- `__pycache__/`

## 数据处理流程

1. 下载 cutoff 原始数据：

```bash
python download_data.py --start_event 226
```

2. 生成原始 HDF5 序列：

```bash
python generate_raw_h5.py
```

默认输出 `datasets/data.hdf5`。

3. 生成训练用预处理数据：

```bash
python preprocess.py
```

默认输出 `datasets/preprocessed.h5`。

## 训练

默认训练 MoE shape 模型：

```bash
python train.py
```

可覆盖常用训练参数：

```bash
python train.py --epochs 20 --batch_size 512 --lr 0.0001 --hidden 128 --model_name moe
```

训练输出：

- `outputs/<run_id>/`：训练曲线、配置快照、验证集可视化
- `saved_models/<model_name>.keras`：服务默认加载的模型文件

当前服务默认模型路径是 `saved_models/moe.keras`，也可以通过 `MODEL_PATH` 环境变量覆盖：

```bash
MODEL_PATH=saved_models/moe.keras python serve.py
```

## 在线服务

启动服务：

```bash
python serve.py
```

默认监听 `0.0.0.0:8000`。

健康检查：

```bash
curl http://localhost:8000/health
```

查询预测：

```bash
curl "http://localhost:8000/predictions/306?server=3&tier=1000"
```

返回字段包含：

- `history`：已缓存的历史切点预测
- `latest_slice`：当前切点的未来曲线预测；活动未开始、已结束或历史长度不足 `t_min` 时可能为 `null`

## 可视化

对已训练模型生成验证集切片可视化：

```bash
python visualize.py --model_path saved_models/moe.keras
```

默认输出到 `outputs/viz_test/val_slice_viz/`。

## 注意事项

- 当前主要面向 Bestdori CN 服务器，默认 server 为 `3`。
- 时间步长固定为 30 分钟。
- `t_min=48`，即活动开始后约 24 小时才会进行预测。
- 在线服务后台每 30 分钟刷新一次活跃活动 cutoff，并按档位更新预测缓存。
- Bestdori 数据版权和使用条件不属于本仓库许可证覆盖范围。

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
