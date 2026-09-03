# Specification: Retraining the Three-Branch Model for FPGA Deployment

## 1. Objective

Redesign and retrain the existing three-branch sEMG classifier with hardware-friendly operators and quantization-aware training (QAT). This is not a simple retraining run: the computational graph must be made implementable on a small FPGA while retaining as much accuracy as possible.

The current model combines:

- Context PLIF branch
- Hybrid ConvLIF/Jaccard branch
- Delay-SNN branch

Dataset and task: NinaPro DB5 Exercise 1, 13 classes. The original FP32 fusion accuracy is 91.10%. A proxy using quantized inputs and weights but floating-point operators reached 90.72%. The model has 696,748 parameters; INT4 Context/Hybrid weights plus INT8 Delay weights require about 349 KiB.

## 2. Target devices and budgets

Primary target: Nexys4 DDR, XC7A100T.

| Device | LUT | FF | BRAM | DSP |
|---|---:|---:|---:|---:|
| XC7A100T | 63,400 | 126,800 | 135 | 240 |
| XC7Z010 (optional portability target) | 17,600 | 35,200 | 60 | 80 |

Suggested implementation budgets:

- XC7A100T: no more than 45k LUT, 100 BRAM, and 180 DSP
- XC7Z010, if attempted: no more than 12k LUT, 50 BRAM, and 60 DSP

## 3. Why the current design does not fit

The previous Vitis/Vivado 2024.2 implementation used 62,531/63,400 LUTs (98.6%) and 133/135 BRAMs, so placement failed. DSP use was 119/240. Timing was +0.19 ns at 50 MHz and -2.26 ns at 100 MHz. Estimated inference latency was roughly 1.9 s at 50 MHz.

The main problem is floating-point computation throughout the datapath. Sharing one GELU `erf` unit did not help: the HLS LUT estimate changed from 37,036 to 37,371. This indicates that the design needs operator and data-format changes, not only operator sharing.

For comparison, the integer Delay-SNN reference achieved about 83% accuracy using 7.6% LUT, 7 BRAM, no DSP, and 100 MHz. Those figures are historical measurements from the deployment project and should not be generalized to the three-branch system.

## 4. Required operator changes

| Original operation | Required hardware-oriented alternative |
|---|---|
| GELU / `erf` | ReLU, ReLU6, or hard-tanh |
| LayerNorm | remove it, replace it with a static affine transform, or fold a hardware-friendly normalization |
| BatchNorm | fold it into the preceding linear or convolution layer |
| Softmax | move it to the host; FPGA returns logits or class scores |
| Jaccard division | shift approximation, reciprocal lookup table, or removal after an ablation |
| Floating membrane update | fixed-point membrane, preferably signed Q8 with `beta = 230/256` or another reported `k/256` value |
| Floating weights/activations | QAT with INT4/INT8 weights and fixed-point activations |

There must be no floating-point division, square root, exponential, or error function in the inference datapath.

## 5. Quantization-aware training

QAT must cover weights, intermediate activations, and membrane state. Use a straight-through estimator for rounding during backpropagation. Recommended starting formats are:

- Context and Hybrid weights: signed INT4, per-output-channel scale
- Delay weights: signed INT8
- Activations and membrane: signed Q8.8 or a documented narrower alternative
- Decay: integer `k/256`

Report the exact saturation range, rounding rule, scale granularity, reset rule, and whether each bias is quantized or kept in a wider accumulator.

## 6. Structural fallback options

If operator replacement and QAT are insufficient, evaluate these changes in order:

1. Reduce hidden dimensions or simulation steps.
2. Share or time-multiplex compute units.
3. Remove the Hybrid branch and use the Context + Delay configuration; the previous balanced two-branch result was about 90.85%.
4. Reduce selected INT4 tensors to INT3 or INT2 only after a validation-set ablation.

All model selection and fusion calibration must use the validation set. The test set is reserved for the final report.

## 7. Export and verification requirements

Deliver all of the following:

- retrained `state_dict`
- packed binary weights plus a machine-readable manifest
- a NumPy-only, bit-accurate fixed-point reference implementation
- 13 to 19 deterministic golden vectors with inputs, intermediate values, logits, and predicted classes
- validation and full independent-test metrics

If the data layout changes, define a new versioned format instead of silently reusing the old manifest. The NumPy reference must reproduce the exported integer path, including clipping, round-to-nearest-even, accumulator widths, membrane update, thresholding, and reset.

## 8. Acceptance criteria

- Fits within the suggested XC7A100T resource budget.
- Strict independent-test accuracy is at least 83%; the target is approximately 90%.
- Python/NumPy and RTL or HLS predictions are bit-exact on every golden vector.
- Target latency is below 100 ms per window at 50–100 MHz.
- The report clearly distinguishes measured synthesis results from estimates.

## 9. Delay-SNN compatibility template

Unless an ablation proves otherwise, retain the known deployment conventions:

- topology: `96 -> 64 -> 128 -> 64 -> 13`
- signed INT8 weights
- signed Q8 membrane state
- `beta = 230/256`
- round-to-nearest-even
- hard reset after a spike
- FPGA returns spike counts or logits; softmax stays on the host
- reference latency: 2,524,541 cycles, or about 25 ms at 100 MHz
- reference resources: 7 BRAM, 7.6% LUT, 0 DSP

## 10. Delivery checklist

- [ ] operator-ablation table
- [ ] QAT configuration and training logs
- [ ] validation-selected checkpoint
- [ ] packed weights and manifest
- [ ] NumPy bit-accurate simulator
- [ ] golden vectors
- [ ] full test-set accuracy, macro-F1, and gesture-only accuracy
- [ ] synthesis, implementation, timing, resource, and latency reports
- [ ] reproducible commands and environment information
