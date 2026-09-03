# 07. Exporting Fixed-Point Weights and Verifying Them with NumPy

QAT still stores trainable tensors as floating-point values. Deployment requires integer codes, scales, format metadata, and an independent implementation that reproduces every inference rule.

## 7.1 Package design

A practical package separates binary data from metadata:

```text
model_weights.npz
model_manifest.json
golden_vectors.npz
```

The manifest should record model version, tensor names and shapes, axis order, bit width, signedness, scale granularity, fixed-point fractional bits, clipping bounds, rounding, accumulator width, membrane decay, threshold, reset rule, and input/output layout.

## 7.2 Quantizing weights for export

```python
import numpy as np

def quantize_weight_per_output(weight, bits=4):
    weight = np.asarray(weight, dtype=np.float32)
    limit = 2 ** (bits - 1) - 1
    axes = tuple(range(1, weight.ndim))
    maximum = np.max(np.abs(weight), axis=axes, keepdims=True)
    scale = np.maximum(maximum / limit, 1e-12)
    codes = np.rint(weight / scale).clip(-limit, limit).astype(np.int8)
    return codes, np.squeeze(scale).astype(np.float32)
```

`np.rint` uses round-to-nearest-even for halfway cases. If RTL uses another rule, implement that exact rule instead.

## 7.3 Folding affine layers

Fold inference-time BatchNorm and hardware affine constants before final quantization whenever this matches the RTL graph. Quantize the folded weight, not the pre-folded weight, because folding changes the dynamic range and therefore the optimal scale.

## 7.4 Export skeleton

```python
import json
import numpy as np

arrays = {}
manifest = {"format_version": 1, "tensors": {}}

for name, weight in exportable_weights.items():
    code, scale = quantize_weight_per_output(weight, bits=4)
    arrays[f"{name}.code"] = code
    arrays[f"{name}.scale"] = scale
    manifest["tensors"][name] = {
        "shape": list(code.shape),
        "bits": 4,
        "signed": True,
        "scale_axis": 0,
    }

np.savez_compressed("model_weights.npz", **arrays)
with open("model_manifest.json", "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
```

Validate every expected key, shape, dtype, and range before writing the package.

## 7.5 NumPy-only inference

The reference must import NumPy, not PyTorch. It should follow the hardware order of operations:

```python
import numpy as np

def rne_shift(value, bits):
    return np.rint(value / float(2 ** bits)).astype(np.int64)

def fixed_lif_step(current, membrane, beta_code, threshold_code):
    leaked = rne_shift(membrane.astype(np.int64) * beta_code, 8)
    updated = leaked + current.astype(np.int64)
    spike = updated >= threshold_code
    updated = np.where(spike, 0, updated)  # hard reset
    return spike.astype(np.int8), updated
```

Real designs also need explicit saturation and accumulator widths. Python integers do not overflow like RTL, so the simulator must model clipping or wraparound deliberately.

## 7.6 Cross-checking QAT and NumPy

```python
torch_logits = run_qat_model(batch)
numpy_logits = run_numpy_reference(batch.cpu().numpy())

difference = np.abs(torch_logits.cpu().numpy() - numpy_logits)
print("argmax agreement", np.mean(torch_logits.argmax(1).cpu().numpy() == numpy_logits.argmax(1)))
print("maximum absolute error", difference.max())
print("mean absolute error", difference.mean())
```

Argmax agreement alone is insufficient. A large logit difference can be hidden when the winning margin happens to be large. Diagnose the first failing sample layer by layer.

## 7.7 Golden vectors

Store a small deterministic set containing difficult cases and all classes where possible:

```python
np.savez_compressed(
    "golden_vectors.npz",
    sample_index=indices,
    input_code=input_codes,
    expected_logits=numpy_logits,
    expected_class=numpy_logits.argmax(1),
    labels=labels,
)
```

Include useful intermediate tensors or checksums. Golden vectors connect the QAT model, NumPy reference, HLS simulation, RTL simulation, and physical board test.

## 7.8 Summary

- Export integer codes and metadata, not merely rounded floating-point tensors.
- Keep the reference independent of the training framework.
- Match rounding, clipping, overflow, decay, threshold, and reset exactly.
- Use golden vectors to localize discrepancies across implementations.

Next: [08-resource-estimate-deployment.md](08-resource-estimate-deployment.md)
