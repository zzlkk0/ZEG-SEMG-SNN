# ZEG-SEMG-SNN

面向 NinaPro DB5 表面肌电（sEMG）13 类手势识别的 SNN 研究与教学代码。

仓库包含从基础 Delay-SNN 复现、Context/Hybrid 多分支优化、SNN-LSTM 与
RF-SNN 对照实验，到硬件友好 QAT、定点参考实现和 FPGA 部署教学的完整代码脉络。

> 本仓库不包含 NinaPro 数据、训练 checkpoint、导出权重或 bitstream。
> 文中准确率是特定数据划分下的软件实验结果，除非明确提供实板证据，否则不能
> 称为 FPGA 实测结果。

## 主要结果

所有主要结果使用 NinaPro DB5 Exercise 1：Rest + 12 个动作，16 通道、200 Hz，
500 ms 窗口、100 ms hop。训练 repetition 为 1/2/4/6，验证为 3，测试为 5。

| 路线 | 测试准确率 | 说明 |
|---|---:|---|
| Delay-SNN | 83.93% | 可学习轴突延迟，纯 PyTorch SNN |
| SNN-LSTM 最佳实验 | 83.84% | 未超过 Delay-SNN |
| RF-SNN 严格划分 | 80.24% | 52 点 RF80 实验 |
| Context + Hybrid + Delay | **91.10%** | 无动作边界、无未来信息的软件 FP32 主结果 |
| 硬件友好 QAT 三分支 | **91.11%** | 定点/查表语义的软件参考结果，不是 RTL/FPGA 测量 |

这是相同受试者的跨 repetition 测试，不是 leave-one-subject-out；测试窗也有
80% 重叠。请同时阅读每个实验目录中的 `RESULTS.md`，不要脱离协议引用数字。

## 仓库结构

```text
docs/
  tutorial/                     从 SNN 基础到 FPGA 部署的 8 章中文教程与 notebooks
  specifications/               硬件友好重训和 Delay 分支接口说明
training/
  semg_snn_fpga_reproduction/   Delay-SNN 论文方法的软件复现
  semg_snn_90_loop/             Context/Hybrid/三分支融合、QAT、定点导出
  semg_snn_lstm/                SNN-LSTM 对照实验
  semg_rf_snn/                  RF-SNN 对照实验
scripts/
  sanitize_notebooks.py         清除 notebook 输出及本机路径
```

## 快速开始

建议 Python 3.10–3.12。PyTorch 的 CUDA 安装方式取决于系统和驱动；若需要 GPU，
优先按 PyTorch 官方安装器选择匹配版本，再安装其余依赖。

```bash
git clone https://github.com/zzlkk0/ZEG-SEMG-SNN.git
cd ZEG-SEMG-SNN

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

先运行不需要数据的语法检查：

```bash
python -m compileall -q training
python scripts/check_public_release.py
```

从教程开始：

```bash
python -m pip install -r requirements-notebooks.txt
jupyter lab
```

打开 [`docs/tutorial/00-README.md`](docs/tutorial/00-README.md)。01–08 章中的
合成数据示例可以独立学习；`docs/tutorial/notebooks/` 的真实项目案例需要按
[`DATASET.md`](DATASET.md) 准备 NinaPro DB5 数据和 checkpoint。

## 训练入口

基础 Delay-SNN：

```bash
cd training/semg_snn_fpga_reproduction
python prepare_db5.py --input-dir data/extracted --output-dir data/processed
python train.py --epochs 100 --run-dir runs/baseline
```

Context/Hybrid 主线：

```bash
cd training/semg_snn_90_loop
python prepare.py
python prepare_continuous_context.py

python train.py \
  --model context_class_adaptive \
  --context 23 \
  --continuous-context \
  --stream-context \
  --epochs 24 \
  --patience 7 \
  --batch-size 256 \
  --lr 0.0002 \
  --run-name context23_class_plif_stream
```

硬件友好 QAT：

```bash
cd training/semg_snn_90_loop
python train_qat.py --help
python export_hw_fixed.py --help
python evaluate_hw_ensemble.py --help
```

每个子工程的 README 给出了更完整的命令、模型初始化关系和评估方式。

## 复现原则

- 只用训练 repetition 计算归一化统计。
- 用验证集选择 checkpoint、融合权重和 Rest bias。
- 测试集只做最终报告，不用来调参。
- 同时报告 accuracy、macro-F1 和非 Rest 手势准确率。
- 明确区分 FP32、QAT/定点参考、HLS/RTL 仿真与实板结果。
- 新实验使用新的 `runs/<name>`，不要覆盖原结果。

## 数据和模型文件

NinaPro 数据及训练生成物未上传。所需目录、文件格式和下载注意事项见
[`DATASET.md`](DATASET.md)。`.gitignore` 会阻止常见数据、checkpoint、权重、
bitstream 和工具构建目录被误提交。

## 文献与限制

Delay-SNN 复现参考：

> M. A. Scrugli, G. Leone, P. Busia, P. Meloni,
> “sEMG-Based Gesture Recognition with Spiking Neural Networks on Low-Power FPGA,”
> DASIP 2024. DOI: 10.1007/978-3-031-62874-0_2.

RF-SNN 的研究背景和复现差异见
[`training/semg_rf_snn/RESEARCH.md`](training/semg_rf_snn/RESEARCH.md)。

本仓库当前没有附带许可证。在许可证明确之前，请勿假定代码可用于商业用途。
