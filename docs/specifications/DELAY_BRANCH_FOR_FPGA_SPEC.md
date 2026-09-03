# Specification: Adding the Delay Branch to the FPGA System

## 1. Existing system

The hardware-friendly Context + Hybrid implementation for Nexys4 DDR uses Q8.8 activations, INT4 weights, spike-gated accumulation, and host-side softmax/fusion. On a 300-sample board test it achieved 88.0% accuracy and matched the NumPy reference bit-for-bit.

Reported implementation figures are approximately 33.6% LUT, 88 BRAM, 11 DSP, 50 MHz, and 655 ms per input. The FPGA returns raw branch logits; softmax and fusion run on the host.

The current two-branch fusion uses weights `[0.625, 0.375]` and rest-class bias `-0.48`. Adding the Delay branch changes the validation-selected fusion to `[0.5, 0.3, 0.2]` with the same `-0.48` rest bias and reaches 91.11% in the software evaluation.

## 2. Task

Export the existing `delay62_finetune/best.pt` checkpoint without retraining, implement a NumPy-only hardware-fixed Delay reference, create golden vectors, and define a board-compatible input/output protocol. The current HLS/RTL datapath for Context and Hybrid must remain compatible.

## 3. Weight package

Export a versioned NPZ package containing at least:

- `layer_codes`: signed INT8 weight codes
- one floating or fixed-point scale per output channel or per documented layer
- quantized biases or documented wider bias values
- topology metadata
- delay arrays for the first three layers and `max_delay = 62`
- decay numerator `k` for `beta = k/256`
- membrane width, fractional bits, thresholds, rounding mode, and reset mode

Reject an export if a tensor has the wrong shape or a delay is outside `[0, 62]`.

## 4. NumPy hardware reference

Provide an implementation named `HWFixedDelay` or equivalent. It must not depend on PyTorch during inference and must model:

- signed INT8 weights
- signed Q8 membrane values
- decay `k/256`
- round-to-nearest-even
- hard reset
- integer delay ring buffers
- layer spike counts and final class counts

The FPGA output is the 13-class count vector. Softmax and fusion remain host operations.

## 5. Input format

Recommended host preprocessing:

```text
raw sEMG [100, 16] -> deterministic delta/event encoding [100, 96]
```

Specify the exact delta algorithm, clipping, thresholding, dtype, and channel order. Pack the event tensor in time-major order with least-significant-bit-first bit packing. Preserve compatibility with the legacy 1,200-byte payload when practical; otherwise introduce an explicit protocol version.

The board must not silently apply a different preprocessing path from the NumPy reference.

## 6. Golden vectors

Extend the existing set of 18 deterministic samples with:

- packed Delay input bytes
- per-layer intermediate spike counts or checksums
- final 13-class count vector
- Delay argmax
- fused probabilities or scores
- final fused prediction and ground-truth label

Each vector must have a stable sample index and be reproducible from the independent test set.

## 7. Host fusion

Use Delay temperature `0.03` and the validation-selected fusion:

```text
p_context = softmax(context_logits)
p_hybrid  = softmax(hybrid_logits)
p_delay   = softmax(delay_counts / 0.03)
p_fused   = 0.5 * p_context + 0.3 * p_hybrid + 0.2 * p_delay
p_fused[rest] *= exp(-0.48)
p_fused   = p_fused / sum(p_fused)
prediction = argmax(p_fused)
```

Clip logits only if necessary for numerical stability and document the clipping rule. Batch requests must preserve the same sample order across all three branches.

## 8. Metrics

Report on the complete independent test set:

- Delay floating-point accuracy
- Delay fixed-point accuracy
- Context + Hybrid accuracy
- uncalibrated three-branch fusion accuracy
- calibrated three-branch fusion accuracy
- macro-F1 and gesture-only accuracy
- bit-exact agreement rate between NumPy and hardware outputs

## 9. Acceptance criteria

- Existing Delay checkpoint exports without retraining.
- NumPy fixed-point inference is deterministic.
- Every golden vector is bit-exact against RTL/HLS output.
- The input protocol is fully specified and versioned.
- Host fusion reproduces the recorded software result within the stated numerical tolerance.
- Resource, timing, latency, and accuracy claims are labeled as measured or estimated.

## 10. Delivery checklist

- [ ] Delay NPZ weights and manifest
- [ ] NumPy-only `HWFixedDelay`
- [ ] pack/unpack utilities and protocol description
- [ ] 18 golden vectors with intermediates
- [ ] host-side three-branch fusion code
- [ ] full independent-test report
- [ ] HLS/RTL integration notes
- [ ] board test and bit-exact comparison
