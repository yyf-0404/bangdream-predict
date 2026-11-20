# bangdream-predict

BanG Dream! 国服活动档线预测。最终模型为 **原条件曲线 `conditional-curve-v1`**。开场满 24 小时、有足够的实测档线后，模型从最近一次共同观测推算各档每半小时的累计分数，直到结算。发布权重随仓库提供。[最终权重](saved_models/conditional-curve.npz) · [模型与验证结果](docs/conditional-curve.md)

## 启动服务

在项目根目录执行以下命令。

```bash
pip install -r requirements.txt
python serve.py
```

服务监听 `0.0.0.0:8000`。查询国服 328 期 1500 档的示例如下。

```bash
curl http://127.0.0.1:8000/health
curl 'http://127.0.0.1:8000/predictions/328?server=3&tier=1500'
```

`/health` 返回加载的模型版本和权重 SHA-256。预测响应中的 `history` 是每次起报时预测的结算分。活动进行中，`latest_slice.prefix_curve` 给出从最新锚点到结算的逐半小时预测。活动结束后 `latest_slice` 为 `null`。历史只使用各次起报时已发布的分数。服务按需下载数据，并每半小时刷新活跃活动。模型权重、算法缓存版本和原始观测共同决定可复用的预测缓存。

## 数据、复现与验证图

下面两步下载国服档线，按固定活动清单生成 `datasets/joint_v2_through_328.h5`。已有冻结快照可直接用于回放。数据源后续补录可能改变重建结果。

```bash
python download_data.py --start_event 226
python generate_joint_h5.py
```

在训练服务器上运行以下命令，可用训练集的 79 场活动拟合原条件曲线。默认输出 `outputs/conditional-curve/conditional-curve.npz`。默认服务始终加载 `saved_models/conditional-curve.npz`。

```bash
python train.py --model-type conditional-curve
```

以下命令用发布权重回放验证集，生成 `outputs/viz_test/index.html`。每张图含实际走势、各次预测的结算分、代表起报时刻的整段预测及结算偏差。

```bash
python visualize.py
```

国服 322–328 七场历史验证中，从满 24 小时开始逐半小时起报，500 档结算误差为 **3.97%**，1500 档为 **8.40%**。对应整段曲线误差为 **1.55%** 和 **2.12%**。[活动划分](data/splits/cn_type_balanced_through_328_v3.json) · [结算标签](data/splits/cn_verified_finals_through_328.json) · [评分口径](docs/conditional-curve.md)

兼容原逐档 MoE 的入口为 `python train.py --model-type moe`，已有调用可通过 `MODEL_PATH=saved_models/moe.keras python serve.py` 显式加载旧权重。

## 许可证

[MIT License](LICENSE)。外部数据的版权与使用条件归其提供方。
