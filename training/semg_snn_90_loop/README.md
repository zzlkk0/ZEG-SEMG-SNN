# NinaPro DB5 13 类 SNN 优化工程

本工程是在原始延迟编码 SNN 的 83.93% 基线上进行的一轮持续优化。最终在同一套
NinaPro DB5 Exercise A 严格重复划分下得到：

- **91.10%**：完全无动作边界、无未来信息的在线流式结果；
- **92.03%**：连续窗口、但已知 repetition 段边界；
- **93.32%**：先过滤有效窗口、允许在空隙处重置状态的段级协议。

最可信的实时部署数字是 **91.10%**。另外两项用于和常见论文预处理协议比较，不能
当作同一在线条件下的结果。详细定义、指标和失败实验见 [RESULTS.md](RESULTS.md)，
文献依据见 [RESEARCH.md](RESEARCH.md)。

## 数据和划分

- 数据集：NinaPro DB5，Exercise A；
- 类别：13 类，即 Rest + 12 个手指动作；
- 采样率：200 Hz，16 通道；
- 窗口：100 点（500 ms），步长 20 点（100 ms），80% 重叠；
- 训练：repetition 1/2/4/6；
- 验证：repetition 3；
- 测试：repetition 5；
- 样本数：44,630 / 11,060 / 11,276；
- 模型选择、融合权重和静息偏置均只使用验证集，测试标签不参与这些计算。

测试集中 Rest 占 63.44%，所以工程同时保存 overall accuracy、macro-F1 和
gesture-only accuracy，不能只看总体准确率。

## 工程位置

```text
[repository root]/
├── semg_snn_fpga_reproduction/   # 原始 83.93% 延迟编码 SNN 工程
└── semg_snn_90_loop/             # 本优化工程
```

数据、代码、检查点和 JSON 结果都在上述工程目录中，没有散放在 home 根目录。

## 已有环境

实验复用了现有环境，没有新建环境：

```text
Python:       python
PyTorch:      2.5.1
CUDA:         12.4
GPU:          NVIDIA GeForce RTX 3080 20 GB
NumPy:        2.2.6
scikit-learn: 1.8.0
SciPy:        1.16.3
```

## 复现最可信的无边界在线路线

```bash
cd training/semg_snn_90_loop
PY=python

# 已生成的数据可直接使用；需要重建时才执行：
$PY prepare.py
$PY prepare_continuous_context.py

# 完全无边界的 2.2 秒 PLIF 上下文模型
$PY train.py \
  --model context_class_adaptive \
  --context 23 \
  --continuous-context \
  --stream-context \
  --epochs 24 \
  --patience 7 \
  --batch-size 256 \
  --lr 0.0002 \
  --run-name context23_class_plif_stream \
  --init runs/context23_class_plif_continuous/best.pt

# 与单窗 ConvLIF/Jaccard SNN、延迟编码 SNN 融合
$PY evaluate_ensemble.py \
  context_class_adaptive_stream,23,runs/context23_class_plif_stream/best.pt \
  hybrid,1,runs/hybrid_sja_v1/best.pt \
  delta,3,training/semg_snn_fpga_reproduction/runs/delay62_finetune/best.pt \
  --output runs/strict_stream_three_expert_metrics.json

# 在全部连续窗口上审计因果平滑；验证集最终会选择 width=1
$PY evaluate_full_stream.py \
  --context 23 \
  --checkpoint runs/context23_class_plif_stream/best.pt \
  --output runs/strict_full_stream_context23_metrics.json \
  --probabilities runs/strict_full_stream_context23_probabilities.npz
```

## 关键文件

```text
prepare.py                         严格划分、原始窗口和 336 维特征
prepare_continuous_context.py      每一个连续窗口的特征及时间索引
model.py                           Feature/Context/PLIF/ConvLIF/Jaccard/混合 SNN
train.py                           训练、早停、流式与边界模式
evaluate_ensemble.py               验证集权重、静息偏置与测试评估
evaluate_full_stream.py            全连续流、无边界的因果平滑审计
evaluate_subject_ensemble.py       个体化校准实验
evaluate_calibrated_fusion.py      65 参数类别级校准实验
runs/*/best.pt                     各实验最佳检查点
runs/*metrics.json                 完整可机器读取的指标和候选参数
```

## 推荐使用

若目标是离线论文对比，可报告 93.32%，但必须明确写出“过滤段并在空隙重置”。
若目标是实时手环或假肢控制，应以 91.10% 为主结果，并进一步在真实连续采集、
未知动作切换时刻和跨天数据上复验。
