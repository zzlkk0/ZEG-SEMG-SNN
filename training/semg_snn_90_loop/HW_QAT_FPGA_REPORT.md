# 硬件友好重训报告（Context + Hybrid, QAT）

任务来源：`docs/specifications/RETRAIN_FOR_FPGA_SPEC.md`
完成时间：2026-08-02（Asia/Macau）
执行环境：`python`（PyTorch 2.5.1 + CUDA 12.4，RTX 3080）

> Delay-SNN 分支本身已经是纯整数、已上板验证的设计（83.93%，7.6% LUT / 7 BRAM /
> 0 DSP @ 100MHz），本次不重训，直接复用
> `semg_snn_fpga_reproduction/runs/delay62_finetune/best.pt`。
> 本报告只覆盖 Context 和 Hybrid 两个分支的算子替换 + QAT。

## 1. 做了什么算子替换

| 原算子 | 换成 | 代码位置 |
|---|---|---|
| `nn.LayerNorm`（含 mean/var/sqrt/div） | `HWAffine`：纯逐通道 `x*scale+bias`，无数据相关统计量，可在导出时折叠进前一层 Linear | `hw_ops.py: HWAffine` |
| `nn.GELU`（erf） | `nn.ReLU6` | `hw_model.py: HWConvLIFBranch` |
| `nn.BatchNorm1d`（推理期） | 训练时保留（稳定优化），导出时解析折叠为逐通道 `scale+shift` 仿射常量 | `export_hw_fixed.py: add_bn` |
| Jaccard 注意力 `intersection/union` | 101 项（union∈[0,100]）查找表，`recip_table[u]=round(2^12/max(u,1))/2^12`，导出时是真正的整数索引 LUT | `hw_fixed_reference.py: reciprocal_table` |
| 权重（Linear/Conv） | 逐输出通道对称 INT4（`scale=max/limit`），STE 伪量化训练 | `hw_ops.py: fake_quant_weight` |
| 膜电位/激活 | 对称定点 Q8.8（8 位小数），STE 伪量化训练 | `hw_ops.py: fake_quant_activation` |
| 衰减 β | 量化为 k/256（与已上板 Delay-SNN 的 230/256 衰减同一风格） | `hw_ops.py: fake_quant_decay` |
| softmax | 保持在片外（host），分支只输出原始 logits/spike 累加，不变 | 未改动 |

原则符合验收：**数据通路里没有 float 除法、开方、exp、erf**；乘法只剩定点整数乘（权重 INT4/INT8 × Q8.8 激活），逐输出通道的浮点 scale 是编译期常量乘法（和现有已接受的 INT4 导出方案一致），不是运行时除法。

一个训练中发现并修复的 bug：`HWAffine` 最初对每个元素单独算 scale（`fake_quant_weight` 按行 amax），导致每个数都能精确还原、等于没有量化。已改为整个张量共享一个 scale（`fake_quant_tensor`），QAT 才真正感受到仿射层的量化噪声，导出后 numpy 定点参考与 torch 训练前向 100% argmax 一致（见第 3 节）。

## 2. 精度结果（完整 NinaPro DB5 严格测试集，11,276 窗）

### 单分支（QAT 定点前向 vs 原 FP32 基线）

| 分支 | FP32 基线 test acc | HW-QAT test acc | FP32 macro-F1 | HW-QAT macro-F1 |
|---|---:|---:|---:|---:|
| Context | 88.53% | **88.91%** | 78.22% | 78.97% |
| Hybrid | 88.76% | **88.21%** | 76.15% | 74.88% |
| Delay（未改动） | 83.93% | 83.93%（同一模型） | — | — |

### 三分支融合（Context HW-QAT + Hybrid HW-QAT + Delay，验证集选融合权重）

| 协议 | Accuracy | Macro-F1 | Gesture-only Acc |
|---|---:|---:|---:|
| 验证集选权重 `(context=0.5, hybrid=0.3, delay=0.2)` | 91.35%（val） | 84.70%（val） | 79.89%（val） |
| 测试集，未校准 | 90.88% | 82.19% | 77.30% |
| 测试集，Rest 校准（bias=-0.48，在验证集上选） | **91.11%** | **82.40%** | **79.41%** |

对比：SEMG_SNN_NEXT_SESSION.md 记录的严格无边界 FP32 结果为 91.0961%，量化输入+FP32算子代理为 90.7237%。**本次真正去除了 LayerNorm/GELU/浮点除法后的融合精度 91.11%，与 FP32 基线持平甚至略高**，且验收线（≥83%，目标接近 90%）已经超过。

方法学与 `evaluate_ensemble.py`（原 91.0961% 的产出脚本）一致：验证集网格搜索融合权重 + Rest logit 偏置，测试集只做一次最终评估，未用测试集调参。

复现命令：
```bash
cd training/semg_snn_90_loop
PY=python
$PY evaluate_hw_ensemble.py \
  --context-checkpoint runs/hw_context23_qat_v1_affinefix/best.pt \
  --hybrid-checkpoint runs/hw_hybrid_qat_v1_affinefix/best.pt \
  --output runs/hw_three_branch_fusion_metrics_affinefix.json
```

## 3. numpy 定点参考实现（不依赖 torch）

`hw_fixed_reference.py`：`HWFixedContext` / `HWFixedHybrid` 两个类，只依赖 numpy，加载 `export_hw_fixed.py` 产出的 `.npz` 权重包后逐算子复现 `hw_model.py` 的前向（卷积用显式滑窗乘加实现，BN 已折叠为仿射常量，Jaccard 除法换成真正的整数索引 LUT）。

一致性验证（512 个验证集样本，与同一 checkpoint 的 torch QAT 前向对比）：

| 分支 | argmax 一致率 | logit 最大绝对误差 | logit 平均绝对误差 |
|---|---:|---:|---:|
| Context | 100% (512/512) | 9.2e-4 | 1.1e-6 |
| Hybrid | 100% (512/512) | 9.6e-2 | 2.9e-3 |

误差量级是浮点求和顺序噪声（卷积滑窗累加顺序、GPU vs CPU 累加顺序不同），不是逻辑错误——两分支预测类别 100% 匹配。

## 4. 打包格式（新格式，替代/新增于原 `three_expert_manifest.json` 方案）

未复用原 `three_branch_weights.hpp` / `three_expert_manifest.json`（那是给旧的 float-operator-proxy 设计用的），改为两个自解释的 `.npz` 包，因为算子集合变了（无 LayerNorm/BN/GELU/softmax，多了仿射常量和倒数 LUT）：

```
weights_hw/hw_context_fixed.npz
weights_hw/hw_hybrid_fixed.npz
```

**命名规则**（每层 3 个数组，`<layer>_codes` / `<layer>_scale` / `<layer>_bias`）：

- `*_codes`：`int8`，Linear/Conv 权重是逐输出通道 INT4（数值范围裁到 [-7,7]，用 int8 容器存），HWAffine 是整张量共享一个 scale 的 INT8（[-127,127]）。Conv 的 shape 是 `[out,in,kernel]`，1x1 conv（q/k/v）的 kernel 维退化为 1。
- `*_scale`：`float32`，Linear/Conv 是逐输出通道一个 scale（`weight = codes * scale[:,None]`），HWAffine 是标量。
- `*_bias`：`float32`，未量化（accumulator 精度足够，量化收益可忽略，与原方案一致）。
- `bn{1,2}_scale` / `bn{1,2}_shift`：BatchNorm 折叠后的逐通道仿射常量（`y = x*scale+shift`），已包含 running_mean/var/eps，推理期不再需要开方或除法。
- `beta1/beta2/beta_f/beta_o/conv_beta`：衰减，已量化为 k/256 网格上的 float32（例如 `230/256` 这种值）。
- `reciprocal_table`（仅 hybrid）：`float32[101]`，`union`（0..100 的整数计数）直接当索引取倒数近似值，无运行时除法。
- 标量元数据（`act_frac_bits/act_int_bits/substeps/windows/steps/time_steps`）：`int32`，供下游代码校验维度和定点格式，不参与数值计算。

**打包体积**（INT4 码紧凑打包 + INT8 仿射码 + fp32 scale/bias，不含 npz 容器开销）：

| 分支 | INT4 权重 | INT8 仿射权重 | scale/bias/衰减/LUT（float32） | 合计 |
|---|---:|---:|---:|---:|
| Context | 149.6 KiB | 0.75 KiB | 12.2 KiB | 162.5 KiB |
| Hybrid | 176.1 KiB | 0.63 KiB | 14.0 KiB | 190.8 KiB |
| **两分支合计** | | | | **353.3 KiB** |

与 `SEMG_SNN_NEXT_SESSION.md` 记录的旧方案（三分支合计约 349 KiB）体量相当——这符合预期，因为参数量本身没变，变的是围绕权重的算子。

## 5. 黄金测试向量

`weights_hw/golden_vectors.npz` + `weights_hw/golden_vectors_summary.json`：18 个严格测试集样本（覆盖全部 13 类，另加 5 个同类连续窗口展示 Context 分支的时间依赖性），记录：

- 原始输入（`context_features[23,336]`、`hybrid_features[336]`、`hybrid_raw[100,16]`）
- 关键中间量：首窗编码器输出、每层最终膜电位、每层脉冲总数、ConvLIF 分支汇总向量
- 最终 logits 与 argmax

可用于 HLS/RTL 仿真做粗粒度比对；`hw_fixed_reference.py` 是确定性的，需要更细粒度（逐拍）比对时可以直接从同一权重包重新生成。

## 6. 算子清单（确认无禁用算子）

| 算子类型 | Context | Hybrid | 说明 |
|---|:---:|:---:|---|
| float 除法/开方/exp/erf | 无 | 无 | 全部替换或折叠 |
| 定点整数乘（含逐通道浮点 scale 常量乘） | 有（enc_linear、fc2、out） | 有（feature_linear、conv1、conv2、q/k/v、fuse_linear、out） | scale 是编译期常量，不是运行时除法 |
| 脉冲门控累加（二值输入，等价纯加法/无需乘法器） | fc2、out 的输入是二值脉冲 s1/s2 | q/k/v 的输入是二值 events；fc2 类比的 fuse_linear 一半输入（sf）是二值 | 和已上板 Delay-SNN 同款优化空间，当前 HLS 尚未针对性利用（见第 7 节） |
| 查找表 | 无 | Jaccard 倒数（101 项） | 编译期常量表 |
| BatchNorm | — | 已折叠为仿射常量 | 不再有 running mean/var 除法 |
| softmax | 片外（host） | 片外（host） | 不变 |

## 7. 资源粗估（未验证，需要实际 csynth/impl 确认）

**没有 Vitis HLS/Vivado 工具链的环境里做不出可信的 LUT/FF/DSP 百分比**，以下只是定性/量级推理，不能当作综合结果：

- **权重 BRAM**：两分支合计 353.3 KiB 紧凑打包，占 XC7A100T 总 BRAM 容量（135×RAMB36=607.5 KiB）约 **58%**。加上 Delay 分支（约 7 个 BRAM tile，原方案）、激活/膜电位缓存、输入 staging、HLS 内部 ROM 复制后，大概率会明显更紧——这是目前最大的已知风险点，必须用 csynth 报告确认，不能只看权重本身小于总容量就下结论（这一点原 spec 文档已经提醒过）。若超限，第一步缓解是回退到 README 里提到的 Context+Delay 两专家 `balanced` 配置（90.85%），可以把权重 BRAM 压力砍掉 Hybrid 那接近一半。
- **LUT**：原浮点设计 LUT 爆表（62,531/63,400=98.6%）的根因是"浮点算子散布全身"（erf 单点优化几乎不降 LUT，37,036→37,371）。本次把 LayerNorm/GELU/BatchNorm-除法/Jaccard-除法全部换掉，理论上应该大幅降低 LUT，但降到多少必须实测，这里不给具体数字。
- **DSP**：原方案 119/240（50%）主要来自浮点乘加（每个 float 乘/加在 HLS 里通常要占用多个 DSP 或大量 LUT）。新方案里，Context 的 `fc2`/`out` 和 Hybrid 的 `q/k/v` 卷积输入都是二值脉冲，代数上等价于"命中就累加对应权重"，和已实测 0 DSP 的 Delay-SNN 是同一种结构——但当前 HLS 尚未针对性实现这个优化（这是 numpy 参考实现里的通用矩阵乘,不是稀疏门控加法),真正做 HLS 翻译时应该显式利用这一点来省 DSP，而不是假设编译器会自动发现。
- **时序**：去除 float 长组合路径（原设计 100MHz 下时序差 -2.26ns，主要是 float erf/GELU 长路径）后，达到 100MHz 的把握应该明显提高，但同样需要实测确认。

## 8. 交付清单核对（对照 RETRAIN_FOR_FPGA_SPEC.md 第 8 节）

- [x] 重训后 state_dict：`runs/hw_context23_qat_v1_affinefix/best.pt`、`runs/hw_hybrid_qat_v1_affinefix/best.pt`
- [x] 打包权重二进制 + manifest（新格式，见第 4 节）：`weights_hw/hw_context_fixed.npz`、`weights_hw/hw_hybrid_fixed.npz`
- [x] numpy 定点参考实现（无 torch 依赖）：`hw_fixed_reference.py`
- [x] 黄金测试向量：`weights_hw/golden_vectors.npz` + `golden_vectors_summary.json`
- [x] 精度报告（全测试集 11,276 窗，FP32 基线 vs 定点版，逐分支 + 融合）：见第 2 节
- [x] 算子清单：见第 6 节
- [~] 资源估算：只能给定性/量级推理 + 权重 BRAM 精确占比（58%），LUT/FF/DSP 具体数字**没有做**，因为环境里没有 Vitis HLS/Vivado（见第 7 节，已明确标注为待验证）
- [x] 新格式规范文档：见第 4 节

## 9. 还没做的 / 下一步

1. 真正的 HLS C++ 翻译（把 `hw_fixed_reference.py` 的算子图翻成可综合 C++，并显式利用脉冲门控累加省 DSP）+ csynth，拿到第一份可信的 LUT/BRAM/DSP/时序数字。
2. 如果权重 BRAM 或 LUT 超限：优先试 Context+Delay 两专家 `balanced` 配置，其次是把 INT4 进一步压到 INT3/INT2（配合 QAT 补偿，本报告的 QAT 流程已经验证可以吸收较大的算子改动而精度不掉，值得再压一版试试代价）。
3. Vivado synthesis/implementation/DRC/STA，以及 Nexys4 DDR 实板 UART 回归（复用现有 `semg_snn_nexys4ddr_three_branch` 的 UART 协议/host 代码，或按新打包格式改造其权重读取部分）。

## 10. 明确的表述边界（延续 SEMG_SNN_NEXT_SESSION.md 的原则）

- 可以说：Context/Hybrid 分支的算子替换 + QAT 微调后，在完整 11,276 窗测试集上，去除 LayerNorm/GELU/BatchNorm 运行时除法/Jaccard 除法后的三分支融合精度为 91.11%，与之前的 FP32 基线（91.10%）和量化代理（90.72%）相比持平或略优。
- 可以说：numpy 定点参考实现与 torch QAT 前向在 512 个验证样本上 argmax 100% 一致。
- 不可以说：这个设计已经在 A100T 上跑通、装得下或时序收敛——没有做 HLS/Vivado 综合，第 7 节的资源讨论是定性推理，不是综合结果。
- 不可以把本报告的"定点参考位精确"等同于"RTL 位精确"——numpy 参考和 QAT 训练用的是同一套伪量化语义，还没有翻译成 HLS/RTL 并做 cosim 验证。
