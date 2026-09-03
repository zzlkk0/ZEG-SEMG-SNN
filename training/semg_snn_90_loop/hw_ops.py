"""Hardware-friendly quantization primitives for FPGA-oriented QAT.

All fake-quant ops use a straight-through estimator (STE): forward rounds to
a fixed-point grid, backward passes the gradient through unchanged.  This
lets training simulate the integer datapath (fixed weight bits, Q-format
activations, shift-friendly decay) that the exported model must run bit
matched on the FPGA, without needing real integer kernels during training.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


round_ste = _RoundSTE.apply


def fake_quant_weight(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-output-channel symmetric fake quantization, dim 0 is output."""

    limit = 2 ** (bits - 1) - 1
    reduce_dims = tuple(range(1, weight.dim()))
    amax = weight.detach().abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-8)
    scale = amax / limit
    q = round_ste(weight / scale).clamp(-limit, limit)
    return q * scale


def fake_quant_tensor(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Single shared-scale symmetric fake quantization over the whole tensor.

    Unlike fake_quant_weight, this does not carve out a scale per output row,
    so a 1-D per-channel affine (HWAffine) actually loses precision instead
    of reconstructing itself exactly (a per-element scale would let every
    value hit the quantization grid boundary and round-trip losslessly).
    """

    limit = 2 ** (bits - 1) - 1
    amax = weight.detach().abs().amax().clamp_min(1e-8)
    scale = amax / limit
    q = round_ste(weight / scale).clamp(-limit, limit)
    return q * scale


def fake_quant_activation(
    x: torch.Tensor, frac_bits: int = 8, int_bits: int = 8
) -> torch.Tensor:
    """Symmetric fixed-point fake quantization, signed Q(int_bits).(frac_bits)."""

    scale = 2.0 ** (-frac_bits)
    limit = 2 ** (int_bits + frac_bits - 1) - 1
    q = round_ste(x / scale).clamp(-limit, limit)
    return q * scale


def fake_quant_decay(beta: torch.Tensor, frac_bits: int = 8) -> torch.Tensor:
    """Quantize a (0,1] decay to k/2**frac_bits, matching a shift-decay RTL."""

    scale = 2.0 ** (-frac_bits)
    limit = 2 ** frac_bits
    q = round_ste(beta / scale).clamp(1, limit)
    return q * scale


def fake_quant_reciprocal(value: torch.Tensor, frac_bits: int = 12) -> torch.Tensor:
    """Quantize a (0,1] reciprocal to the grid a small LUT would store."""

    scale = 2.0 ** (-frac_bits)
    limit = 2 ** frac_bits
    q = round_ste(value / scale).clamp(0, limit)
    return q * scale


class HWAffine(nn.Module):
    """Per-channel affine replacement for LayerNorm.

    Unlike LayerNorm this has no data-dependent mean/variance, so it is an
    exact elementwise scale+bias that can be folded into the preceding
    Linear/Conv weight and bias at export time (same algebra as BatchNorm
    folding).  Keeping it as a separate module during QAT lets the fake-quant
    activation step between "linear" and "affine" see the same numeric range
    the folded, quantized weight will produce.
    """

    def __init__(self, dim: int, bits: int = 8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.bits = bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = fake_quant_tensor(self.weight, self.bits)
        return x * weight + self.bias


class QuantLinear(nn.Linear):
    def __init__(self, *args, bits: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bits = bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = fake_quant_weight(self.weight, self.bits)
        return F.linear(x, weight, self.bias)


class QuantConv1d(nn.Conv1d):
    def __init__(self, *args, bits: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bits = bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = fake_quant_weight(self.weight, self.bits)
        return F.conv1d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
