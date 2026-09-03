# Delay 分支定点交付说明（补齐三分支上板，交接给训练服务器）

> 自包含任务说明。承接你之前完成的 `HW_QAT_FPGA_REPORT.md`（Context/Hybrid 的
> QAT 硬件友好重训）。现在部署端已把 **Context+Hybrid 两分支成功跑在 Nexys4
> 实板上**（详见下文“当前状态”）。本任务只需你补交 **Delay 分支的定点交付
> 物**，格式与 Context/Hybrid 完全对齐，部署端据此写 HLS、加进现有核心、做三
> 分支融合，目标复现报告里的 **91.11%**。

## 0. 一句话任务

把 Delay 分支（`semg_snn_fpga_reproduction/runs/delay62_finetune/best.pt`，即参
与 91.11% 融合的那个整数 Delay-SNN）导出成一套**与 Context/Hybrid 同风格的定点
交付物**：定点权重 npz + 无 torch 的 numpy 定点参考 + 黄金向量 + 精度报告 + 输入
规范。Delay 本来就是整数网络，所以**不需要重训**，只需按下述格式**导出和封装**，
并明确它的输入和融合方式。

## 1. 当前部署状态（你需要对齐的目标）

部署端已完成（Vitis HLS / Vivado 2024.2，Nexys4 DDR / XC7A100T）：

- Context+Hybrid 合并核心：Q8.8 定点、INT4 权重、脉冲门控累加，**实板 300 样本
  88.0%，与 `hw_fixed_reference.py` 的 numpy 定点参考逐位一致**。
- 资源 LUT 33.6% / BRAM 88% / DSP 11%，50 MHz，单次 655 ms。
- FPGA 只输出**每分支的原始值**（context 13 logits、hybrid 13 logits），
  **softmax + 融合 + argmax 全在主机**。
- 当前两分支融合权重 [0.625, 0.375]（把三分支的 [0.5,0.3] 重归一化），Rest
  bias −0.48。加上 Delay 后应恢复三分支 [0.5, 0.3, 0.2] + bias −0.48 → 91.11%。

Delay 加进来后，FPGA 会额外输出 Delay 分支的 **13 个 spike 计数**（不做 softmax），
主机按融合公式合成三分支。

## 2. 需要交付的内容（每一项都必需）

### 2.1 定点权重包 `weights_hw/hw_delay_fixed.npz`

风格对齐 `hw_context_fixed.npz` / `hw_hybrid_fixed.npz`（部署端已能解析这套约定）：

- 每层权重：`<layer>_codes`（`int8` 容器；若是 per-output 对称 INT8 则值域
  [-127,127]，若 INT4 则 [-7,7]），`<layer>_scale`（`float32`，逐输出通道，
  `weight = codes * scale[:,None]`），`<layer>_bias`（`float32`，未量化）。
- Delay 网络拓扑 96→64→128→64→13，四个 Linear/synapse 层。给出每层 codes/scale/
  bias 及形状。
- **整数轴突延迟**：每个前三层神经元一个整数延迟，给成 `<layer>_delays`
  （`int` 数组，值域 [0, max_delay]），并给 `delay_depth` / `max_delay` 标量。
- 衰减 β：量化为 k/256 网格的 `float32`（例如 230/256），标量或逐层。
- 膜电位定点格式：给 `mem_frac_bits` / `mem_int_bits`（Delay 用的是 signed Q8
  还是别的，明确位宽），阈值 `threshold_q`（各层，定点整数）。
- 标量元数据：`time_steps`（100）、`input_channels`（96）、`num_classes`（13）、
  `decay_numerator`（如 230）、`fraction_bits`（如 8），`int32`，供部署端校验。

如果这套约定和 Delay 的整数语义不完全契合，请**新增字段并在报告里写清规范**
（每个数组名、形状、dtype、量化/打包方式、字节语义），不要为了套格式而丢信息。

### 2.2 numpy 定点参考 `hw_fixed_reference.py` 里新增 `HWFixedDelay`

- **只依赖 numpy，不依赖 torch**，与现有 `HWFixedContext`/`HWFixedHybrid` 并列。
- 逐算子复现 Delay-SNN 的整数前向：signed Q8 膜电位、k/256 移位衰减、
  round-to-nearest-even、阈值发放后 hard reset、整数轴突延迟环、每类 spike 计数、
  时间步循环。
- `infer(...)` 返回 **13 个 spike 计数**（以及可选 argmax）。这 13 个计数就是喂给
  三分支融合的原始量，**FPGA 只输出它，不在片上做 softmax**。
- 关键：必须和它将要实现的 FPGA 定点算术**位精确**（部署端会写位精确 HLS 去
  对齐它，就像 Context/Hybrid 那样）。

### 2.3 Delay 输入规范（最关键，务必写清）

Delay 分支吃的是 **100×96 的二值 delta 事件**（time-major）。请明确：

1. **这 96 通道的 delta 事件如何从 `data/test.npz` 的 `raw`（[100,16] int8）得到**：
   - 是主机端预计算（像旧树莓派/Nexys4 pipeline 那样，FPGA 直接收现成 delta），
     还是需要 FPGA 做在线 delta 编码？
   - 如果是主机预计算：给出**确定性的 numpy 函数**（raw[100,16] → delta[100,96]
     二值），部署端主机端照做后打包进 UART payload。
   - 如果需要片上编码：给出精确的定点编码算法（阈值、状态、位宽）。
   - 强烈建议**主机端预计算**（和现有 Context/Hybrid 输入一样由主机准备），这样
     FPGA 只做推理。请按这个方向给 numpy 函数。
2. delta 事件的 bit 打包顺序（time-major、每字节 LSB-first？），与旧
   `semg_snn_nexys4ddr_vivado` 的 1,200-byte 格式是否一致（如一致直接说明复用）。

### 2.4 黄金向量 `weights_hw/golden_vectors.npz` 增补 Delay 字段

- 复用**当前 18 个样本**（保持和 Context/Hybrid 黄金同一批样本、同一顺序），新增：
  - `delay_input`（每样本的 100×96 二值 delta 事件，或其打包字节）
  - `delay_spike_counts`（13）、`delay_argmax`
  - 若干中间量（每层膜电位/脉冲总数）便于逐层比对
- 这样部署端能把三分支黄金放在一起，端到端验证融合。

### 2.5 三分支融合规范（确认 + 给公式）

报告里是：每分支 softmax → 概率，加权和 [0.5,0.3,0.2]，log 后对 Rest(0) 加
bias −0.48，argmax。其中 Delay 的概率是 `softmax(spike_counts / temperature)`，
`temperature=0.03`（即 `(counts/100)/0.03` 之类，请给**精确公式**：counts 是原始
计数还是除以 time_steps 后的比例，temperature 作用在哪一步）。

请在报告里写死：Delay 概率 = `softmax( f(spike_counts) )` 的 `f` 到底是什么，
以及最终三分支 `combined = 0.5*p_ctx + 0.3*p_hyb + 0.2*p_delay`、
`logp=log(clip(combined))`、`logp[0]+=-0.48`、`argmax` 是否与你产出 91.11% 的
`evaluate_hw_ensemble.py` 完全一致。

### 2.6 批量输入构建（测代表性精度用）

部署端已有 `make_test_batch.py`（用 EMGDataset 为一批测试样本构建 context[23,336]、
hybrid[336]、raw[100,16]、truth）。请补充**同样 index 下 Delay 分支的输入**
（delay delta 事件 [100,96]），最好给一个函数或说明，让部署端能为同一批 300 个
测试样本同时准备三分支输入，从而在实板上测三分支融合精度并对齐 91.11%。

### 2.7 精度报告

- 全测试集 11,276 窗：Delay 单分支定点精度、三分支融合（含 Rest 校准）精度、
  macro-F1。确认与报告 91.11% 一致。
- numpy 定点参考 vs 原 torch/整数前向 的 argmax 一致率（像 Context/Hybrid 那样）。

## 3. 验收标准

1. `HWFixedDelay` 无 torch 依赖，`infer` 返回 13 spike 计数，与其定点算术自洽。
2. 三分支融合（Context QAT + Hybrid QAT + 本 Delay）在全测试集 = 报告的 91.11%
   （或据实报告实际值，不粉饰）。
3. delay 输入的 numpy 预处理函数确定性、可复现，部署端主机可直接调用。
4. 黄金向量含 Delay 字段，与 Context/Hybrid 同批样本对齐，可端到端验证。
5. Delay 网络算子清单确认无 float 除法/开方/exp/erf（softmax 在主机不算）。

## 4. 回传清单

- [ ] `weights_hw/hw_delay_fixed.npz`（+ 若格式有新增，写规范）
- [ ] `hw_fixed_reference.py` 增补 `HWFixedDelay`（无 torch）
- [ ] delay 输入的 numpy 预处理函数（raw → 100×96 delta 事件）+ 打包顺序说明
- [ ] `golden_vectors.npz` 增补 delay 字段（同 18 样本）
- [ ] 三分支融合精确公式 + 参数（temperature、weights、rest bias）
- [ ] 为 `make_test_batch.py` 的同一批样本产出 delay 输入的方法
- [ ] 精度报告（Delay 单分支 + 三分支融合，全测试集）

## 5. 备注

- FPGA 上 Delay 分支只需输出 13 个 spike 计数（整数），不做 softmax/除法。
- 部署端会把 Delay 的定点算术写成位精确 HLS（signed Q8、k/256 移位衰减、整数
  延迟环、脉冲计数），和你的 `HWFixedDelay` 对齐——所以 `HWFixedDelay` 必须是
  **可位精确复现的确定性整数实现**，不能用浮点近似。
- 已上板的 `semg_snn_nexys4ddr_vivado`（整数 Delay-SNN，83%，7 BRAM/0 DSP）可作
  为算子/格式参考，但请以 `delay62_finetune` 这个**参与 91.11% 融合的具体权重**
  为准导出；如果两者其实是同一模型，请在报告里说明并可直接复用其 RTL 语义。

宁可保守准确，不要为凑 91.11% 而模糊输入或融合细节。
