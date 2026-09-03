# 06. Hardware-Friendly Operators and Quantization-Aware Training

This chapter turns a convenient floating-point training graph into an integer-oriented inference graph. Fake quantization keeps tensors in floating-point storage during training but forces forward values onto the same grids used by hardware.

## 6.1 Round with an STE

```python
import torch

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

round_ste = RoundSTE.apply
```

The forward pass sees quantization error. The backward pass treats rounding as the identity so optimization can continue.

## 6.2 Per-output-channel symmetric weight quantization

```python
def fake_quant_weight(weight, bits=4, eps=1e-8):
    limit = 2 ** (bits - 1) - 1
    reduce_dims = tuple(range(1, weight.ndim))
    maximum = weight.detach().abs().amax(dim=reduce_dims, keepdim=True)
    scale = (maximum / limit).clamp_min(eps)
    code = round_ste(weight / scale).clamp(-limit, limit)
    return code * scale
```

For `Linear`, output channels are rows. For `Conv1d`, output channels are the first dimension. Signed INT4 uses codes `[-7, 7]` in this symmetric convention; state the convention because some implementations use `[-8, 7]`.

Use the quantized weight in the operator rather than modifying the stored parameter:

```python
import torch.nn.functional as F

def quantized_linear(x, layer, bits=4):
    return F.linear(x, fake_quant_weight(layer.weight, bits), layer.bias)
```

## 6.3 Fixed-point activation quantization

```python
def fake_quant_activation(x, fractional_bits=8, total_bits=16):
    scale = float(2 ** fractional_bits)
    lower = -(2 ** (total_bits - 1))
    upper = 2 ** (total_bits - 1) - 1
    code = round_ste(x * scale).clamp(lower, upper)
    return code / scale
```

Apply it after affine operations and to membrane state wherever the hardware stores a quantized value. Quantizing weights alone does not reproduce an integer datapath.

## 6.4 Quantized decay

Constrain decay to a shift-friendly denominator:

```python
def fake_quant_decay(beta, fractional_bits=8):
    denominator = float(2 ** fractional_bits)
    return round_ste(beta * denominator).clamp(0, denominator - 1) / denominator
```

At inference, `beta = k/256` becomes an integer multiplication followed by an eight-bit right shift, with an explicitly defined rounding rule.

## 6.5 Replacing LayerNorm

LayerNorm requires input-dependent mean, variance, square root, and division. It cannot be made cheap by quantizing only its parameters. Replace it with a data-independent affine transform:

```python
class HWAffine(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(channels))
        self.bias = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        return x * self.weight + self.bias
```

This transform can later be folded into the preceding linear layer. It is not mathematically equivalent to LayerNorm; QAT fine-tuning is what recovers performance.

## 6.6 Replacing GELU

Use `ReLU6` when bounded nonlinearity helps control the activation range:

```python
activation = torch.nn.ReLU6()
```

ReLU, hard-tanh, and a piecewise-linear approximation are alternatives. Select them through a validation-set ablation.

## 6.7 Folding BatchNorm

For inference-time BatchNorm parameters `gamma`, `beta`, running mean `mu`, and variance `var`:

```text
scale = gamma / sqrt(var + epsilon)
shift = beta - mu * scale
y = x * scale + shift
```

If `x = Wx + b`, fold the constants as:

```text
W_folded[o] = scale[o] * W[o]
b_folded[o] = scale[o] * b[o] + shift[o]
```

The square root is evaluated once during export, not in the FPGA datapath.

## 6.8 Replacing division with a lookup table

For Jaccard attention, the union count is an integer in a small range. Precompute reciprocal values:

```python
def reciprocal_table(max_union=100, fractional_bits=12):
    denominator = 2 ** fractional_bits
    table = []
    for union in range(max_union + 1):
        divisor = max(union, 1)
        table.append(round(denominator / divisor))
    return torch.tensor(table, dtype=torch.int32)

score_q12 = intersection * reciprocal_q12[union]
```

This converts runtime division into integer indexing and multiplication. Verify the finite-table approximation against the QAT graph.

## 6.9 QAT fine-tuning

```python
model = HardwareFriendlyModel()
model.load_compatible_fp32_weights(fp32_checkpoint, strict=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

for epoch in range(qat_epochs):
    model.train()
    for x, y in train_loader:
        logits = model(x)  # fake quantization is active inside forward()
        loss = torch.nn.functional.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    validate_and_save_best(model, validation_loader)
```

Start from a compatible FP32 checkpoint, use a smaller learning rate than the original training run, and select the checkpoint on the validation set.

## 6.10 Verify that quantization is real

A common bug is applying per-output-channel quantization after adding a singleton dimension. The reduction then sees one value at a time, chooses that value as its own scale, and reconstructs it almost exactly.

Check the forward effect directly:

```python
with torch.no_grad():
    quantized = fake_quant_weight(weight, bits=4)
    error = (weight - quantized).abs()
    print("max error", error.max().item())
    print("mean error", error.mean().item())
    print("unique normalized codes", torch.unique(torch.round(quantized / scale)).numel())
```

Also inspect code ranges, saturation rates, activation histograms, and the difference between QAT and NumPy logits. A configuration flag is not evidence that quantization is taking effect.

## 6.11 Summary

- Replace data-dependent transcendental and normalization operations before QAT.
- Quantize weights, activations, membrane state, and decay consistently.
- Fold static affine transforms at export.
- Validate the numerical effect of every fake-quantizer.

Next: [07-export-numpy-verify.md](07-export-numpy-verify.md)
