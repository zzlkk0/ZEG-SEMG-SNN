# 从零开始:脉冲神经网络(SNN)训练到 FPGA 部署 —— 教学课件

这是一套面向"会用 PyTorch,但没做过 SNN / 没做过硬件量化部署"的学习者的分步教程。
课程内容基于一个真实项目的经验总结:一个多分支 sEMG(表面肌电)手势识别 SNN,
从最初的浮点训练一路做到可在 Xilinx Artix-7 FPGA(Nexys4 DDR 开发板)上跑的
纯定点、无浮点运算的硬件设计。所有代码示例在本教程里都改写成了**自包含的合成数据 demo**,
不依赖任何私有数据集,你可以直接抄下来跑。

## 你会学到什么

1. 脉冲神经元(LIF)的基本原理和"替代梯度"训练技巧
2. 如何从零训练第一个能跑起来的 SNN 分类器
3. 如何把 SNN 做深、做多步,让它更接近实用精度
4. 为什么真实系统喜欢用"多分支并行 + 融合"而不是单个大模型,以及怎么做融合
5. 为什么浮点算子(GELU、LayerNorm、Softmax、除法)在 FPGA 上是灾难,量化的动机是什么
6. 怎么把每个浮点算子换成硬件友好的等价物,并用量化感知训练(QAT)找回精度
7. 怎么导出一份不依赖 PyTorch、纯 numpy 的定点参考实现,并验证它和训练时"位对位"一致
8. 怎么做资源(LUT/BRAM/DSP)的诚实估算,以及部署到真实开发板之前还差什么

## 章节列表

| 文件 | 主题 | 你会写的代码 |
|---|---|---|
| [01-snn-basics.md](01-snn-basics.md) | LIF 神经元与替代梯度 | 单个神经元的膜电位仿真 |
| [02-first-training.md](02-first-training.md) | 训练第一个 SNN 分类器 | 单层 LIF + 速率编码,在合成数据上训练 |
| [03-deeper-multistep.md](03-deeper-multistep.md) | 多层、多步、可学习衰减 | 两层 LIF,早停,梯度裁剪 |
| [04-parallel-branches-fusion.md](04-parallel-branches-fusion.md) | 多分支并行模型与融合 | 两个视角不同的分支 + 验证集网格搜索融合权重 |
| [05-quantization-motivation.md](05-quantization-motivation.md) | 为什么要量化 | 无代码,概念+资源预算计算 |
| [06-hw-friendly-ops-qat.md](06-hw-friendly-ops-qat.md) | 硬件友好算子替换 + QAT | 伪量化算子库,替换 LayerNorm/GELU/BN,QAT 训练循环 |
| [07-export-numpy-verify.md](07-export-numpy-verify.md) | 导出定点权重 + numpy 参考实现 | 权重打包、numpy 推理引擎、逐样本比对 |
| [08-resource-estimate-deployment.md](08-resource-estimate-deployment.md) | 资源估算与部署清单 | BRAM 计算脚本 + 诚实边界声明 |

## 学习路径建议

```
01 → 02 → 03                  (纯算法阶段:能训练出一个还不错的 SNN)
         ↓
        04                    (系统阶段:多分支融合把精度做上去)
         ↓
05 → 06 → 07 → 08             (部署阶段:量化、导出、验证、估算)
```

如果你只关心"怎么训练 SNN",看完 01~04 就够了。
如果你的目标就是"怎么把训练好的模型塞进 FPGA",02~03 可以粗读,重点在 05~08。

## 前置知识

- 会用 PyTorch 写基本的训练循环(`optimizer.zero_grad()` / `loss.backward()` / `optimizer.step()` 这个级别)
- 知道什么是交叉熵损失、验证集/测试集划分
- 不需要任何 FPGA/硬件背景,05 章会从零讲起

## 环境

```bash
pip install torch numpy scikit-learn
```

不需要 GPU,本教程所有示例数据量很小,CPU 上几秒到几十秒就能跑完。

## 配套 notebook:同一套流程在真实项目上的完整案例

[`notebooks/`](notebooks/) 目录下有 8 个 Jupyter notebook，把本教程的方法论真正应用
在了产出这些经验的真实项目上——三个结构不同的分支（Context / Hybrid / Delay-SNN）
各自独立的"训练"+"量化"notebook，最后两个 notebook 做融合。用的是真实的 sEMG
手势识别项目代码（`semg_snn_90_loop` + `semg_snn_fpga_reproduction`）和真实
NinaPro DB5 数据，不是合成数据：

| Notebook | 对应章节 | 内容 |
|---|---|---|
| [01_context_training.ipynb](notebooks/01_context_training.ipynb) | 01~03 | 训练 Context 分支（PLIF 上下文 SNN），两阶段课程学习 |
| [02_context_quantization.ipynb](notebooks/02_context_quantization.ipynb) | 05~07 | Context 分支 HW-QAT、量化生效检查、numpy 定点参考实现交叉验证 |
| [03_hybrid_training.ipynb](notebooks/03_hybrid_training.ipynb) | 01~04 | 训练 Hybrid 分支（ConvLIF + Spiking Jaccard Attention）|
| [04_hybrid_quantization.ipynb](notebooks/04_hybrid_quantization.ipynb) | 06~07 | Hybrid 分支 HW-QAT，含 BatchNorm 折叠、Jaccard 除法→查找表 |
| [05_delay_training.ipynb](notebooks/05_delay_training.ipynb) | 01~03 | 训练 Delay-SNN 分支，两阶段：无延迟基线 → 可学习轴突延迟微调 |
| [06_delay_quantization.ipynb](notebooks/06_delay_quantization.ipynb) | 05,08 | Delay-SNN 的训练后量化（PTQ，不需要重新训练），对比真实上板硬件数字 |
| [07_fusion_fp32.ipynb](notebooks/07_fusion_fp32.ipynb) | 04 | 三个 FP32 分支融合，复现 91.10% |
| [08_fusion_hw_qat.ipynb](notebooks/08_fusion_hw_qat.ipynb) | 04,08 | 三个量化后分支融合复现 91.11%、资源估算 |

```
01 Context 训练 ──→ 02 Context 量化 ─┐
03 Hybrid  训练 ──→ 04 Hybrid  量化 ─┼─→ 07 FP32 融合 (91.10%)
05 Delay   训练 ──→ 06 Delay   量化 ─┤    08 HW-QAT 融合 (91.11%)
                                     ┘
```

需要 GPU 和这台机器上已经准备好的项目数据（`semg_snn_90_loop/data/`、
`semg_snn_fpga_reproduction/data/processed/`）才能跑。首次运行建议把训练相关 cell
里的 `epochs` 调小做一次 smoke test，确认能跑通，再用 notebook 里给出的完整
超参数跑出接近参考值的结果。三个分支的量化方式刻意不同（Context/Hybrid 用 QAT，
Delay-SNN 用更简单的 PTQ），这个对比本身就是一个值得注意的教学点——不是所有分支
都需要同一套量化重型武器。

## 一个重要的方法论提醒(贯穿全程)

> **验证集选模型,测试集只在最后报告一次。**
> **量化"看起来生效"不代表真的生效——一定要检查量化后实际用到的编码值范围/精度是否比原始浮点变窄了,否则你可能像本课程 06 章的案例一样,写出一个"看起来在量化、实际上无损重建"的 bug。**

这两条踩过坑,后面章节会具体展开为什么。
