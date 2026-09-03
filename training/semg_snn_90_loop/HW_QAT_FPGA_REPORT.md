# Hardware-Friendly QAT Report: Context + Hybrid

Completed on 2026-08-02 with PyTorch 2.5.1, CUDA 12.4, and an NVIDIA RTX 3080. This report covers the Context and Hybrid branches; the Delay branch was kept unchanged.

## 1. Operator replacements

| Original operator | Replacement | Implementation |
|---|---|---|
| `nn.LayerNorm` with mean/variance/square root/division | `HWAffine`: per-channel `x * scale + bias`, independent of input statistics and foldable at export | `hw_ops.py: HWAffine` |
| `nn.GELU` / `erf` | `nn.ReLU6` | `hw_model.py: HWConvLIFBranch` |
| inference `nn.BatchNorm1d` | retained during optimization, then folded into static per-channel scale and shift | `export_hw_fixed.py: add_bn` |
| Jaccard `intersection / union` | 101-entry reciprocal table for `union` in `[0, 100]`, with `round(2^12 / max(union, 1))` | `hw_fixed_reference.py: reciprocal_table` |
| Linear/Conv weights | per-output-channel symmetric INT4 with STE fake quantization | `hw_ops.py: fake_quant_weight` |
| membrane and activations | symmetric Q8.8 fake quantization | `hw_ops.py: fake_quant_activation` |
| decay `beta` | `k/256` grid | `hw_ops.py: fake_quant_decay` |
| softmax | host-side; branches return raw logits or accumulated spikes | unchanged |

An early `HWAffine` implementation accidentally quantized each scalar with its own scale, making fake quantization effectively lossless. It was corrected to use a shared scale so the forward pass experiences real quantization error.

## 2. Accuracy on the complete strict test set

Dataset: NinaPro DB5 Exercise 1, 11,276 independent test windows.

### 2.1 Individual branches

| Branch | FP32 accuracy | HW-QAT accuracy | FP32 macro-F1 | HW-QAT macro-F1 |
|---|---:|---:|---:|---:|
| Context | 88.53% | **88.91%** | 78.22% | 78.97% |
| Hybrid | 88.76% | **88.21%** | 76.15% | 74.88% |
| Delay, unchanged | 83.93% | 83.93% | not reported | not reported |

### 2.2 Three-branch fusion

The validation set selected weights `(context=0.5, hybrid=0.3, delay=0.2)`.

| Protocol | Accuracy | Macro-F1 | Gesture-only accuracy |
|---|---:|---:|---:|
| validation set with selected weights | 91.35% | 84.70% | 79.89% |
| independent test, uncalibrated | 90.88% | 82.19% | 77.30% |
| independent test, rest bias `-0.48` selected on validation | **91.11%** | **82.40%** | **79.41%** |

The calibrated result is produced by the saved fusion configuration, not by tuning on the test set.

## 3. NumPy fixed-point reference

The reference inference path depends on NumPy only. A 512-sample cross-check against the PyTorch QAT forward pass produced:

| Branch | Argmax agreement | Maximum absolute logit error | Mean absolute logit error |
|---|---:|---:|---:|
| Context | 100% (512/512) | 9.2e-4 | 1.1e-6 |
| Hybrid | 100% (512/512) | 9.6e-2 | 2.9e-3 |

The Hybrid difference is larger because several quantized stages and the reciprocal lookup accumulate approximation error. It did not change argmax on this 512-sample set, but this is not a proof of bit-exact equivalence on all inputs.

## 4. Export package

The new package supersedes the earlier `three_expert_manifest.json` convention:

```text
export/
├── context_hw_fixed.npz
├── hybrid_hw_fixed.npz
├── hw_fixed_manifest.json
└── golden_vectors.npz
```

It records packed INT4 weights, INT8 affine parameters where applicable, per-output scales, biases, quantized decay values, reciprocal tables, tensor shapes, fixed-point formats, and preprocessing metadata.

| Branch | INT4 weights | INT8 affine weights | scales/biases/decay/LUT | Total |
|---|---:|---:|---:|---:|
| Context | 149.6 KiB | 0.75 KiB | 12.2 KiB | 162.5 KiB |
| Hybrid | 176.1 KiB | 0.63 KiB | 14.0 KiB | 190.8 KiB |
| **Combined** | | | | **353.3 KiB** |

## 5. Golden vectors

The export contains 18 deterministic test samples. Each record includes sample index, quantized input, branch logits, expected class, ground-truth label, and selected intermediate data. The vectors are intended for the NumPy, HLS, RTL, and physical-board comparison chain.

## 6. Inference operator inventory

| Operator type | Context | Hybrid | Note |
|---|:---:|:---:|---|
| floating division, square root, `exp`, or `erf` | none | none | replaced, folded, or moved to host |
| fixed/integer multiplication | present | present | scales are compile-time constants, not runtime division |
| spike-gated accumulation | `fc2` and `out` | event-driven `q/k/v`; binary half of `fuse_linear` input | same optimization opportunity as Delay-SNN |
| lookup table | none | 101-entry Jaccard reciprocal | compile-time constant |
| BatchNorm | none | folded affine constants | no running-statistics computation |
| softmax | host | host | unchanged |

## 7. Resource assessment

The Context and Hybrid packages occupy 353.3 KiB, about 58% of the XC7A100T raw BRAM capacity before banking, buffering, and fragmentation. This leaves limited headroom for activations, Delay weights, FIFOs, and protocol buffers.

The earlier floating-point implementation used approximately 98.6% LUT and failed placement. Removing transcendental functions and runtime division should reduce logic pressure, but the current generic HLS design does not yet exploit all spike-gated additions or optimal compute sharing. Therefore no final LUT, FF, DSP, frequency, or latency claim can be made until C synthesis and Vivado implementation are rerun.

## 8. Deliverables completed

- [x] hardware-friendly Context and Hybrid model definitions
- [x] QAT training and evaluation scripts
- [x] validation-selected checkpoints
- [x] export of packed quantized weights and metadata
- [x] NumPy-only fixed-point reference
- [x] 18 golden vectors
- [x] complete strict-test metrics and calibrated fusion result
- [x] operator inventory and preliminary memory calculation

## 9. Remaining work

- [ ] implement or optimize the three-branch HLS/RTL datapath
- [ ] exploit binary spike-gated accumulation explicitly
- [ ] run HLS C simulation against all golden vectors
- [ ] run synthesis and review inferred arithmetic
- [ ] complete place-and-route and timing closure
- [ ] measure board latency and accuracy
- [ ] consider branch time-multiplexing or lower bit widths if BRAM or LUT use is too high

## 10. Claim boundaries

- The accuracy figures above are measured software results on the specified split.
- The NumPy comparison is measured on 512 samples and establishes argmax agreement, not universal RTL bit-exactness.
- The 353.3 KiB figure is calculated package storage, not post-placement BRAM utilization.
- FPGA resource and timing improvements remain expectations until implementation reports exist.
