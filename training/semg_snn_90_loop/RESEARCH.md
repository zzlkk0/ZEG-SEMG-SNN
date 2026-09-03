# SNN + sEMG 联网研究与路线映射

检索日期：2026-07-28。以下优先列论文主页、作者稿或官方代码，数值只有在数据集、
类别数、划分和窗口协议一致时才能直接比较。

## 直接相关工作

### SpGesture：Spiking Jaccard Attention

- 论文：[NeurIPS 2024 论文页](https://papers.nips.cc/paper_files/paper/2024/hash/409334f42cbb57d07aa152f2d0433ec7-Abstract-Conference.html)
- 预印本：[arXiv:2405.14398](https://arxiv.org/abs/2405.14398)
- 代码：[guoweiyu/SpGesture](https://github.com/guoweiyu/SpGesture)

核心是 ConvLIF 编码/提取、通道级 Spiking Jaccard Attention 和 LIF 分类器。
论文在自采姿态漂移数据上报告 89.26%，并非 NinaPro DB5，不能直接和本工程比较。
本工程据此实现了 `ConvLIFBranch` 和 Jaccard 交并比注意力；单窗模型从 Feature-SNN
的 87.41% 提升到 88.76%，与上下文模型融合后贡献更大。

### 低功耗 FPGA sEMG SNN

- 2024 工作：[sEMG-based gesture recognition with SNNs on low-power FPGA](https://iris.unica.it/handle/11584/469172)
- 后续作者稿：[Real-Time sEMG Processing with SNNs on a Low-Power 5K-LUT FPGA](https://iris.unica.it/retrieve/33ced2bf-a130-4a5e-8bdd-671978405242/Real-Time_sEMG_Processing_with_Spiking_Neural_Networks_on_a_Low-Power_5K-LUT_FPGA.pdf)
- 官方训练仓库：[EOLAB SNN-sEMG-GestureClassification-ForceTracking](https://github.com/eolabcolab/SNN-sEMG-GestureClassification-ForceTracking)

2024 版本在 DB5 报告 85.6%；后续作者稿在 12 个动作上报告 83.17%，并给出 FPGA
功耗/能耗结果。其价值主要是硬件可实现性，而不是最高精度。本工程保留了一个约
23.6k 参数的 Delay-SNN 作为低功耗专家，但主模型尚未量化或综合。

### 可学习时间常数 PLIF

- 论文：[Incorporating Learnable Membrane Time Constant to Enhance Learning of SNNs](https://arxiv.org/abs/2007.05785)

PLIF 说明可学习膜时间常数能提升直接训练 SNN 的时序表达。本工程进一步实现：

- 每层可学习衰减；
- 神经元级衰减偏移；
- 13 个类别各自的历史衰减。

类别/神经元异质时间常数使已知边界连续单模型从 89.03% 提到 89.42%，并提高融合
动作类准确率。

### Temporal Efficient Training（TET）

- 论文：[ICLR 2022 / arXiv:2202.11946](https://arxiv.org/abs/2202.11946)
- 官方代码：[brain-intelligence-lab/temporal_efficient_training](https://github.com/brain-intelligence-lab/temporal_efficient_training)

TET 对所有时刻的损失进行更一致的优化，可缓解只监督最终输出的问题。本轮没有完整
移植 TET，因为当前 Context-SNN 的监督与窗口级加权读出绑定；这是下一轮最值得尝试
的训练目标之一。

### Adaptive multi-delta 与 TAD-LIF

- 论文：[High-speed Low-consumption sEMG-based Transient-state micro-Gesture Recognition](https://arxiv.org/abs/2403.06998)

论文提出 adaptive multi-delta 编码和 TAD-LIF，两个自采数据集报告 83.85% 与
93.52%。本轮做了一个 adaptive-delta 初版，但 DB5 验证约 80.5%，没有超过已有
延迟编码，因此停止；这不等价于完整复现论文。

### Spiking Transformer / ESTU

- 作者稿：[ESTU: Enabling Spiking Transformers on Ultra-Low-Power FPGAs](https://iris.unica.it/retrieve/e43af17f-5826-4407-850d-ff4a9f89bcff/ESTU_Enabling_Spiking_Transformers_on_Ultra-Low-Power_FPGAs%20%281%29.pdf)
- 代码：[EOLAB-2025/ESTU](https://github.com/EOLAB-2025/ESTU)

ESTU 在 DB5 上报告约 87.21%。它更适合下一步做“精度—功耗—资源”联合优化，
当前 91% 方案若直接部署，计算量会高于 FPGA Delay-SNN。

### Spiking Reservoir

- 论文记录：[Event-driven physical reservoir computing for sEMG](https://eprints.gla.ac.uk/360949/)
- DOI：[10.1109/TAI.2025.3592899](https://doi.org/10.1109/TAI.2025.3592899)

旋转神经元 reservoir 的全脉冲方案报告约 80.3%，优势是近传感器、低延迟和硬件
友好，而不是 DB5 的最高精度。若最终目标是极低功耗，可作为复杂 Context-SNN 的
替代基线。

## 非 SNN 上限与可迁移思路

这些工作用于判断“90% 是否合理”，不是与本实验直接公平比较。

- [sEMGXCM](https://www.mdpi.com/2306-5354/10/9/1101)：在 DB5 全 53 类的
  subject-specific 设置报告约 92.3%/94.2%，说明 90% 并非数据本身的硬上限。
- [STCNet](https://doi.org/10.1016/j.compbiomed.2024.109525)：
  时空交叉网络与 subject-aware contrastive learning；官方代码
  [KNU-BrainAI/STCNet](https://github.com/KNU-BrainAI/STCNet)。
- [sEMG attention simple model](https://arxiv.org/abs/2006.03645)：
  表明简单注意力和规范预处理也能在多个 NinaPro 任务上形成强基线。
- [2026 few-shot prototype adaptation](https://www.nature.com/articles/s41598-026-40352-6)：
  DB5 subject-specific 报告约 97.6%，提示少量用户校准很有价值，但协议与本工程
  不同。本轮逐受试者权重校准并未超过全局融合，说明简单校准不足以替代原型适配。

## 本轮从文献到实现的路线

```text
原始 Delay/Delta SNN（83.93%）
        │
        ├── 手工时频特征 + Feature-SNN
        │
        ├── ConvLIF + Spiking Jaccard Attention
        │
        ├── 因果 Context-SNN
        │       └── PLIF 神经元/类别异质时间常数
        │
        └── 验证集概率融合 + 单一 Rest 偏置
                ├── 无边界在线：91.10%
                ├── 已知边界：92.03%
                └── 过滤段：93.32%
```

## 下一步优先级

1. **先冻结协议**：以 `--stream-context` 为唯一在线协议，禁止 repetition 边界和
   过滤窗口平滑。
2. **TET/逐时刻辅助损失**：让 PLIF 在过渡初期也得到直接监督，减少 2.2 秒历史的
   动作切换污染。
3. **动作检测与分类联合 SNN**：单独预测 Rest/transition，再进行 12 类手势分类，
   但动作检测必须只用过去信息。
4. **对比学习预训练**：借鉴 STCNet，在训练 repetitions 上做 subject-aware
   supervised contrastive learning，再将表示蒸馏到 SNN。
5. **未知受试者与跨天测试**：增加 LOSO、跨 session 和电极位移测试，避免同用户
   repetition 结果被误解为泛化能力。
6. **ANN 教师到 SNN 蒸馏**：用 sEMGXCM/STCNet 类教师给 PLIF/ConvLIF 学习软标签，
   推理端仍保持 SNN。
7. **量化与能耗**：统一输入编码，做 INT8/定点、脉冲稀疏率、MAC/AC、延迟和 FPGA
   资源报告；当前准确率不能直接代表低功耗优势。
