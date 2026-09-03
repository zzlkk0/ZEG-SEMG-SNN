# 面向 FPGA 的重训任务说明（交接给训练服务器）

> 这份文档是一个**自包含**的任务说明。执行者（你，服务器上的 Claude）没有
> 之前 FPGA 部署会话的上下文，本文提供你需要的全部背景、实测数据、目标和交
> 付格式。请先完整读完再动手。

## 0. 一句话任务

现有的三分支 sEMG 手势识别模型（91% 精度）**无法部署到目标 FPGA**，因为它
的推理用的是浮点算子，占满了 FPGA 的逻辑资源。你的任务是**用硬件友好的算子
重新设计并重训这个模型（含量化感知训练 QAT）**，在尽量保住精度的前提下，让
它能真正装进目标 FPGA 并实时运行，然后按指定格式导出权重和定点参考实现。

这不是"把同样的模型再训一遍"——那样 FPGA 问题一点不会变。核心是**换算子 +
QAT**。

## 1. 背景：模型与现状

- 模型：Context PLIF-SNN + Hybrid ConvLIF/Jaccard-SNN + Delay-SNN 三分支融合。
- 任务：NinaPro DB5 Exercise 1，13 类手势（含 Rest）。
- 当前精度：FP32 参考 91.10%；量化输入/权重 + FP32 算子的代理 90.72%。
- 参数量：696,748；权重已量化为 per-output 对称 INT4（context/hybrid）、
  INT8（delay），打包二进制约 349 KB。
- 模型定义见 `scripts/three_expert_models.py`（259 行）。
- 导出脚本见 `scripts/export_three_expert.py`（635 行）。
- 训练好的 state_dict：`assets/source_models/{context23,hybrid,delta}_state.pt`。

**关键**：权重量化（INT4/INT8）这部分**没问题**，FPGA 上权重只占约 133/135
的 BRAM，卡边能过。问题**不在权重，在算子**。

## 2. 目标硬件与实测预算（必须满足）

主目标板：**Nexys4 DDR，器件 XC7A100T-CSG324-1**。
次目标板（可选，更小）：**EBAZ4205，器件 XC7Z010CLG400-1**。

| 资源 | XC7A100T（主）| XC7Z010（次）|
|---|---:|---:|
| LUT | 63,400 | 17,600 |
| FF（寄存器）| 126,800 | 35,200 |
| BRAM（RAMB36 tile）| 135 | 60 |
| DSP | 240 | 80 |

**布局器需要余量**：利用率超过约 90% 就会布局失败。因此实际设计目标应留出
余地：

| 资源 | A100T 设计上限（建议 ≤70%）| 理由 |
|---|---:|---|
| LUT | **≤ 45,000** | 现状 62,531（98.6%）导致 Place 失败 |
| BRAM tile | ≤ 100 | 现状 133/135，太满 |
| DSP | ≤ 180 | 现状 119，尚可 |

如果还想同时上 EBAZ（Z010），LUT 要压到 **≤ 12,000**、DSP ≤ 60、BRAM ≤ 50。
这很激进，优先保证 A100T；Z010 作为附加目标。

## 3. 现状诊断：为什么装不下（实测，不是估算）

我在 Vitis HLS 2024.2 + Vivado 2024.2 上完整综合/实现过当前模型（HLS C 综合、
IP 导出、板级实现都跑了），结论：

- **LUT 62,531 / 63,400 = 98.6%，Vivado 布局失败**（`Placer 30-99 Design
  utilization is very high`）。
- BRAM 133/135 = 98.5%，卡边。
- DSP 119/240 = 50%，够。
- 时序：50MHz 下能过（+0.19ns）；100MHz 过不了（−2.26ns，FP erf/GELU 长路径）。
- 单次推理延迟约 1.9 秒 @50MHz——即使能装下也太慢，不实用。

LUT 爆掉的根因：**浮点非线性算子和浮点乘加散布在整个数据通路**，被流水线
复制成多份。已验证：把 GELU 的 erf 改成单实例共享，LUT 几乎不降（37,036→
37,371 HLS 估）——说明不是单一热点，是"浮点算术遍布全身"。

作为对照：同数据集上一个**纯整数设计的 Delay-SNN**（单分支，83%）在同一块
A100T 上只占 **7.6% LUT、7 个 BRAM、0 DSP、100MHz**，轻松上板并已实测位精确
跑通。这就是"硬件友好"的样子，是你要对标的模板。

## 4. 要做什么：硬件友好重新设计 + QAT

### 4.1 算子替换（把"贵"的换成"便宜"的）

| 现在（贵，吃 LUT）| 换成（便宜）| 说明 |
|---|---|---|
| GELU（`nn.GELU`/erf）| ReLU 或 hardtanh / ReLU6 | erf 是最贵的单个算子；ReLU 近乎免费 |
| LayerNorm（`nn.LayerNorm`，含 sqrt+除法）| 移除，或换成定点友好的归一化（如按 2 的幂缩放的 RMSNorm，或训练时折叠进权重）| 除法和开方都很贵 |
| 片上 softmax（exp）| **FPGA 上只做 argmax，softmax 移到主机**| Delay-SNN 就是这么干；FPGA 返回整数计数/logits，PC 算概率 |
| Jaccard 注意力的除法 | 用移位近似，或改成无除法的注意力/直接去掉该分支的除法 | sum(min)/max(sum(max),1) 里的除法很贵 |
| float 膜电位 / LIF 状态 | 定点整数（如 Q8，参考 Delay-SNN：signed Q8、beta=230/256、移位衰减）| 这是能否上板的核心 |
| BatchNorm（推理期）| 折叠进前一层的权重和 bias（推理时 BN 是仿射，可合并）| 推理不应留独立 BN |
| float 权重 | 保持 INT4/INT8，但用 **QAT** 训练 | 见 4.2 |

原则：**推理数据通路里不应出现 float 除法、开方、exp、erf**。加、乘、移位、
比较、查表是可以的。非线性一律用小查找表（LUT）+ 线性插值实现。

### 4.2 量化感知训练（QAT）

- 训练时插入 fake-quant，前向按 INT8/INT4 权重 + 定点激活模拟，反向用 STE。
- 激活也要量化到定点（不是只量化权重）。膜电位、累加器定好位宽和小数位。
- 目标是：训练收敛后，**定点前向 == 浮点前向（在容忍误差内）**，这样导出到
  FPGA 不掉精度。
- 建议 PyTorch 的 `torch.ao.quantization`（QAT 流程）或手写 fake-quant 模块。

### 4.3 结构层面的取舍（如果算子替换后仍超预算）

按对精度影响从小到大依次尝试：

1. 缩小隐藏维度（context 512/256、hybrid 384/256 等可减半试）。
2. 减少时间步 / substep 数。
3. 三分支裁剪：若 Hybrid 分支（含注意力、最贵）性价比低，可只保留
   Context + Delay 两分支融合（README 里 `balanced` 配置就是两专家，90.85%）。
4. 权重进一步降位宽（INT4→INT3/INT2），配合 QAT 补偿。

每一步都要在验证集上量化精度损失，记录到报告里。

## 5. 导出格式（必须与现有 FPGA 流水兼容，或明确新格式）

现有 HLS 读的是 `weights/three_branch_weights.hpp`（constexpr 数组）+ 打包
二进制 `weights/three_expert_default.bin` + `weights/three_expert_manifest.json`
（每个矩阵有 `binary_offset`/`binary_bytes`/`bits`/`scale_key`）。量化方式见
`scripts/export_three_expert.py` 的 `quantize`/`pack_int4`：per-output 对称，
`scale = max/limit`，`q = clip(round(w/scale), -limit, limit)`。

**你导出时请产出以下内容**（放在 `weights/` 下，命名可沿用或新增 `_v2`）：

1. 重训后的 state_dict（`.pt`）。
2. 打包权重二进制 + manifest（沿用现有格式，含每矩阵 offset/bits/scale）。
3. **定点参考实现（关键）**：一个**只用 numpy、不依赖 torch** 的 Python 类，
   逐算子按 FPGA 将要实现的定点算术执行推理（膜电位、查表非线性、移位衰减、
   argmax）。FPGA 必须与它**位精确**一致。参考现有
   `host/proxy_model.py` 的结构，但把 FP 算子换成定点。
4. **黄金测试向量**：至少 13～19 个样本的（输入 → 每层中间值 → 最终 logits/
   argmax）定点参考输出，供 FPGA 仿真逐位比对。
5. 精度报告：在完整 NinaPro DB5 严格测试集（11,276 窗）上的定点模型精度、
   macro-F1、每类准确率。同时报告 FP32 基线和定点版的差距。

如果你改了网络结构导致导出格式必须变，请在报告里**明确写出新格式的规范**
（每个数组的名字、形状、dtype、量化 scale、打包方式、字节布局），因为下游要
据此改写 HLS 的权重读取和 matvec。

## 6. 验收标准

一个"成功"的交付需要同时满足：

1. **能装下**：定点模型的算子清单里无 float 除法/开方/exp/erf；据算子和维度
   估算，A100T 上 LUT ≤ 45k、BRAM ≤ 100 tile、DSP ≤ 180（留布局余量）。
2. **精度可接受**：定点全测试集精度**不低于 83%**（否则不如已上板的 Delay-SNN），
   目标尽量接近 90%。报告实际数值，不粉饰。
3. **位精确可验证**：提供 numpy 定点参考 + 黄金向量，FPGA 可逐位对齐。
4. **实时**：结构上支持毫秒级单次推理（避免 1.9 秒那种），即数据通路的
   总操作数可在 50–100MHz、100ms 内完成。

## 7. 环境（服务器端你应已具备）

- PyTorch + CUDA（部署端确认过 torch 2.5.1 + CUDA 可用）。
- 训练数据集：NinaPro DB5 Exercise 1 处理后数据（**本 FPGA 包内没有**，在你
  的训练工程里，README_CN 提到需要 `data/processed/test.npz` 等）。
- 模型定义可复用 `scripts/three_expert_models.py`；训练循环用你原有的。

## 8. 交付回传给部署端的清单

请把以下打成一个压缩包回传，我（部署端）会据此重做 HLS + 上板：

- [ ] 重训后 state_dict（`.pt`）
- [ ] 打包权重二进制 + manifest（含格式规范说明）
- [ ] numpy 定点参考实现（无 torch 依赖）
- [ ] 黄金测试向量（输入 + 逐层 + 最终输出）
- [ ] 精度报告（全测试集，FP32 基线 vs 定点版，每类）
- [ ] 算子清单 + 资源估算（LUT/BRAM/DSP 粗估，证明留了布局余量）
- [ ] 若结构/格式有变：新格式规范文档

## 9. 参考：能上板的模板

对标 `semg_snn_nexys4ddr_vivado`（同数据集的纯整数 Delay-SNN，83%，已实测在
Nexys4 和 EBAZ 两块板位精确跑通）：

- 每层对称 INT8 权重；signed Q8 膜电位；beta = 230/256 移位衰减；
  round-to-nearest-even；发放后 hard reset；argmax 在 FPGA，softmax 在主机。
- 单窗 2,524,541 cycles @ 100MHz ≈ 25 ms；7 BRAM、7.6% LUT、0 DSP。

让三分支尽量往这个"整数、无除法、非线性查表、softmax 上移"的形态靠。

---

有疑问请在报告里列出假设。宁可保守（能上板的 85%）也不要激进（装不下的 91%）。
