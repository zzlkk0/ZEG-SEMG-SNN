# 08. Resource Estimation and Deployment Checklist

The question "will it fit?" has several different answers. Weight memory can be calculated exactly from tensor shapes and bit widths. LUT, FF, DSP, routing, clock frequency, and latency require synthesis or implementation because scheduling and architecture dominate them.

## 8.1 Operator inventory

| Original operator | Hardware-oriented form | Status |
|---|---|---|
| GELU / `erf` | ReLU6 | no transcendental function |
| LayerNorm | data-independent `HWAffine` | no online mean, variance, square root, or division |
| inference BatchNorm | folded into convolution/linear constants | no independent BatchNorm datapath |
| Softmax | host-side | FPGA returns logits or spike counts |
| attention division | integer reciprocal lookup table | no runtime divider |

Search the exported graph and HLS/RTL source as well as the PyTorch model. An operator removed from one representation may still survive in another.

## 8.2 Exact weight-memory calculation

```python
from math import ceil

def tensor_storage_bytes(shape, bits):
    elements = 1
    for dimension in shape:
        elements *= dimension
    return ceil(elements * bits / 8)

tensors = [
    ("fc1", (256, 96), 4),
    ("fc2", (128, 256), 4),
    ("out", (13, 128), 4),
]

total = sum(tensor_storage_bytes(shape, bits) for _, shape, bits in tensors)
print(total, "bytes")
```

Convert bytes to BRAM tiles only after accounting for port width, depth, fragmentation, double buffering, and whether scale/bias tables share a memory. A theoretical bit total is a lower bound, not a placed-memory count.

For the QAT Context + Hybrid package in this project, stored model data total approximately 353.3 KiB: 162.5 KiB for Context and 190.8 KiB for Hybrid. That is about 58% of the XC7A100T's raw BRAM capacity before layout inefficiency and buffers.

## 8.3 What cannot be inferred reliably from parameter count

- LUT use depends on arithmetic widths, control, multiplexing, unrolling, and routing.
- DSP use depends on whether multiplies are parallel, time-multiplexed, constant-folded, or implemented in LUTs.
- FF use depends on pipeline depth and the amount of live state.
- BRAM use depends on banking and port requirements, not only total bits.
- clock frequency requires post-route timing.
- latency requires the scheduled architecture and measured cycle count.

Use analytical estimates to reject obviously impossible designs. Use HLS synthesis, RTL synthesis, place-and-route, and board measurements for claims.

## 8.4 Resource-reduction controls

If a design does not fit, consider:

1. lower weight or activation precision after a QAT ablation
2. smaller hidden dimensions or fewer simulation steps
3. reuse one compute engine across layers or branches
4. gated accumulation for binary spikes
5. streamed weights when latency and external-memory bandwidth permit it
6. removal of a branch whose marginal fusion gain is small
7. narrower accumulators justified by measured ranges

Every change trades accuracy, latency, bandwidth, or implementation complexity. Re-evaluate the independent test set only after selecting the design on validation data.

## 8.5 End-to-end deployment checklist

- [ ] freeze preprocessing and input layout
- [ ] freeze model topology, quantization, rounding, overflow, and reset rules
- [ ] export versioned weights and manifest
- [ ] pass NumPy versus QAT comparisons
- [ ] create representative golden vectors
- [ ] pass HLS C simulation or RTL simulation bit-exactly
- [ ] synthesize and review LUT, FF, BRAM, DSP, and inferred operators
- [ ] implement and close timing at the target clock
- [ ] measure cycles and end-to-end latency
- [ ] test the same golden vectors on the board
- [ ] run a larger board sample and compare classes with NumPy
- [ ] document host-side softmax, fusion, and calibration
- [ ] label all numbers as calculated, synthesized, implemented, or measured

## 8.6 Evidence boundaries

Use precise language:

- "calculated weight size" for byte arithmetic
- "HLS estimate" for pre-RTL tool reports
- "post-synthesis utilization" after synthesis
- "post-implementation timing" after placement and routing
- "board-measured latency/accuracy/power" only after physical measurement

Do not describe an estimate as proof that a model fits or meets timing.

## 8.7 Course summary

You have followed the complete path from LIF dynamics and surrogate-gradient training through multi-branch fusion, hardware-oriented QAT, integer export, NumPy verification, and deployment planning. The durable workflow is to co-design the model and hardware, validate every numerical transformation, and preserve an honest boundary between software accuracy and implemented hardware evidence.
