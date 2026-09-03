# 05. Why Quantization Is Necessary

## 5.1 Why use an FPGA?

SNNs use binary, often sparse events. An FPGA can exploit that structure with gated accumulation, predictable latency, and low power. This is attractive for wearable or edge sEMG recognition, but a small FPGA has strict compute and on-chip-memory limits.

## 5.2 The FPGA resource ledger

| Resource | Purpose | Typical consumers |
|---|---|---|
| LUT | combinational logic | adders, comparators, shifts, control, approximations |
| FF | registered state | pipelines, state machines, membrane values |
| BRAM | on-chip memory | weights, lookup tables, buffers |
| DSP | hard multiply-accumulate | dense matrix and convolution products |

The XC7A100T on Nexys4 DDR provides about 63.4k LUTs, 135 BRAM tiles, and 240 DSP blocks. A direct floating-point translation of a PyTorch graph can exhaust these resources quickly.

## 5.3 Expensive floating-point operators

| Operator | Hardware cost |
|---|---|
| GELU / `erf` | polynomial or table approximation and substantial control logic |
| LayerNorm | online mean, variance, reciprocal square root, and division |
| Softmax | exponentials and normalization division |
| runtime division | a large variable-latency or deeply pipelined datapath |

In this project, a floating-point three-branch implementation used 62,531 of 63,400 LUTs (98.6%) and failed placement. The lesson is that operator choice can dominate parameter count.

## 5.4 PTQ versus QAT

- Post-training quantization (PTQ) rounds a trained model without further optimization. It is simple but can lose substantial accuracy at INT4.
- Quantization-aware training (QAT) applies fake quantization in the forward pass and uses an STE in the backward pass. Starting from an FP32 checkpoint and fine-tuning for a few epochs lets the model adapt to the quantization error.

The project's INT4-weight, Q8.8-activation QAT system reached 91.11% after fusion, compared with 91.10% for the original FP32 fusion. This is an observed project result, not a general guarantee that quantization improves accuracy.

## 5.5 Fixed-point notation

In this repository, Q8.8 means a signed value with eight fractional bits:

```text
real_value = integer_code / 256
resolution = 1 / 256 = 0.00390625
```

```python
def quantize_fixed(x, fractional_bits=8, total_bits=16):
    scale = 2 ** fractional_bits
    lower = -(2 ** (total_bits - 1))
    upper = 2 ** (total_bits - 1) - 1
    code = round(x * scale)
    code = max(lower, min(upper, code))
    return code / scale
```

Always specify whether the sign bit is included in the reported width; fixed-point naming conventions vary.

## 5.6 Weight and activation granularity

Per-output-channel symmetric weight scales usually preserve accuracy better than one scale for an entire matrix. Weights are static, so the additional scale table is small. Activations change for every sample; a single fixed Q-format is easier to implement and compose across operators.

## 5.7 Deployment sequence

1. Replace LayerNorm, GELU, Softmax, and variable division.
2. Fine-tune the modified graph with QAT.
3. Export integer codes and explicit scales.
4. Verify the complete fixed-point path with a PyTorch-independent NumPy implementation.
5. Estimate memory, then run synthesis and implementation for actual LUT, FF, BRAM, DSP, timing, and latency results.

Next: [06-hw-friendly-ops-qat.md](06-hw-friendly-ops-qat.md)
