"""Hardware-friendly QAT variants of the Context and Hybrid experts.

Design intent (see ../../docs/specifications/RETRAIN_FOR_FPGA_SPEC.md):
  - No float division / sqrt / exp / erf anywhere in the datapath.
  - LayerNorm -> HWAffine (per-channel scale+bias, no data-dependent stats;
    foldable into the preceding Linear at export, like BatchNorm folding).
  - GELU -> ReLU6 (bounded, quantization-friendly, ~free in hardware).
  - BatchNorm1d is kept during training (stabilizes optimization) and is
    exactly foldable into the preceding Conv1d at export time; no separate
    hardware cost once folded.
  - Jaccard attention division -> quantized reciprocal (a small LUT in
    hardware, simulated here as a fake-quantized float reciprocal so
    training sees the same rounding grid the LUT will produce).
  - Weights: per-output symmetric fake-quant (default INT4, matching the
    deployed Context/Hybrid weight format).
  - Activations / membrane potentials: fixed-point fake-quant to a Q(int).(frac)
    grid (default Q8.8), matching the Delay-SNN's Q8 membrane convention.
  - Decay (beta): quantized to k/256, matching the Delay-SNN's 230/256
    shift-decay so all three branches share one hardware decay pattern.

Softmax stays off-chip (branch outputs remain raw logits / spike counts);
argmax and fusion happen on the host, unchanged from the existing pipeline.
"""

from __future__ import annotations

import torch
from torch import nn

from hw_ops import (
    HWAffine,
    QuantConv1d,
    QuantLinear,
    fake_quant_activation,
    fake_quant_decay,
    fake_quant_reciprocal,
)
from model import spike


class HWClassAdaptiveContextSNN(nn.Module):
    """Hardware-friendly PLIF context expert (drop-in for ClassAdaptiveContextSNN)."""

    def __init__(
        self,
        features: int = 336,
        hidden1: int = 512,
        hidden2: int = 256,
        substeps: int = 3,
        weight_bits: int = 4,
        act_frac_bits: int = 8,
        act_int_bits: int = 8,
    ) -> None:
        super().__init__()
        self.enc_linear = QuantLinear(features, hidden1, bits=weight_bits)
        self.enc_affine = HWAffine(hidden1)
        self.fc2 = QuantLinear(hidden1, hidden2, bits=weight_bits)
        self.norm2 = HWAffine(hidden2)
        self.out = QuantLinear(hidden2, 13, bits=weight_bits)
        self.substeps = substeps
        self.act_frac_bits = act_frac_bits
        self.act_int_bits = act_int_bits
        self.beta1 = nn.Parameter(torch.tensor(2.2))
        self.beta2 = nn.Parameter(torch.tensor(2.2))
        self.context_gamma_logit = nn.Parameter(torch.full((13,), 1.4))
        self.beta1_offset = nn.Parameter(torch.zeros(hidden1))
        self.beta2_offset = nn.Parameter(torch.zeros(hidden2))

    def _q(self, x: torch.Tensor) -> torch.Tensor:
        return fake_quant_activation(x, self.act_frac_bits, self.act_int_bits)

    def forward(self, features, raw=None, subject=None):
        if features.ndim == 2:
            features = features[:, None]
        batch, windows, _ = features.shape
        m1 = features.new_zeros(batch, self.enc_linear.out_features)
        m2 = features.new_zeros(batch, self.fc2.out_features)
        output = features.new_zeros(batch, 13)
        r1 = features.new_zeros(())
        r2 = features.new_zeros(())
        b1 = fake_quant_decay(
            torch.sigmoid(self.beta1 + 0.5 * torch.tanh(self.beta1_offset)),
            self.act_frac_bits,
        )[None]
        b2 = fake_quant_decay(
            torch.sigmoid(self.beta2 + 0.5 * torch.tanh(self.beta2_offset)),
            self.act_frac_bits,
        )[None]
        gamma = torch.sigmoid(self.context_gamma_logit)
        lags = torch.arange(windows - 1, -1, -1, device=features.device)
        weights = gamma[None].pow(lags[:, None])
        total_steps = windows * self.substeps
        for window in range(windows):
            current1 = self._q(self.enc_affine(self.enc_linear(features[:, window])))
            for _ in range(self.substeps):
                m1 = self._q(b1 * m1 + current1)
                s1 = spike(m1 - 1.0)
                m1 = m1 - s1
                current2 = self._q(self.norm2(self.fc2(s1)))
                m2 = self._q(b2 * m2 + current2)
                s2 = spike(m2 - 1.0)
                m2 = m2 - s2
                output = output + weights[window] * self.out(s2)
                r1 += s1.mean()
                r2 += s2.mean()
        normalization = self.substeps * weights.sum(dim=0)
        return output / normalization, [r1 / total_steps, r2 / total_steps]


class HWConvLIFBranch(nn.Module):
    """Hardware-friendly ConvLIF + binary Jaccard attention branch."""

    def __init__(
        self,
        channels: int = 128,
        weight_bits: int = 4,
        act_frac_bits: int = 8,
        act_int_bits: int = 8,
        recip_frac_bits: int = 12,
    ) -> None:
        super().__init__()
        self.conv1 = QuantConv1d(16, 64, 7, padding=3, bits=weight_bits)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.ReLU6()
        self.conv2 = QuantConv1d(64, channels, 5, padding=2, bits=weight_bits)
        self.bn2 = nn.BatchNorm1d(channels)
        self.q = QuantConv1d(channels, channels, 1, bits=weight_bits)
        self.k = QuantConv1d(channels, channels, 1, bits=weight_bits)
        self.v = QuantConv1d(channels, channels, 1, bits=weight_bits)
        self.beta = nn.Parameter(torch.tensor(2.2))
        self.act_frac_bits = act_frac_bits
        self.act_int_bits = act_int_bits
        self.recip_frac_bits = recip_frac_bits

    def _q(self, x: torch.Tensor) -> torch.Tensor:
        return fake_quant_activation(x, self.act_frac_bits, self.act_int_bits)

    def forward(self, raw: torch.Tensor):
        x = raw.transpose(1, 2)
        x = self.act1(self.bn1(self.conv1(x)))
        current = self.bn2(self.conv2(x))
        membrane = torch.zeros_like(current[:, :, 0])
        beta = fake_quant_decay(torch.sigmoid(self.beta), self.act_frac_bits)
        events = []
        for time_index in range(current.shape[-1]):
            membrane = self._q(beta * membrane + current[:, :, time_index])
            e = spike(membrane - 1.0)
            membrane = membrane - e
            events.append(e)
        events = torch.stack(events, dim=-1)
        query = spike(self.q(events) - 0.5)
        key = spike(self.k(events) - 0.5)
        value = spike(self.v(events) - 0.5)
        intersection = torch.minimum(query, key).sum(dim=-1)
        union = torch.maximum(query, key).sum(dim=-1).clamp_min(1.0)
        reciprocal = fake_quant_reciprocal(1.0 / union, self.recip_frac_bits)
        attention = (intersection * reciprocal).unsqueeze(-1)
        attended = value * attention + events
        return attended.mean(dim=-1), [events.mean(), query.mean(), value.mean()]


class HWHybridSNN(nn.Module):
    """Hardware-friendly drop-in for HybridSNN (feature current + ConvLIF/Jaccard)."""

    def __init__(
        self,
        features: int = 336,
        hidden: int = 384,
        conv_channels: int = 128,
        steps: int = 12,
        weight_bits: int = 4,
        act_frac_bits: int = 8,
        act_int_bits: int = 8,
    ) -> None:
        super().__init__()
        self.feature_linear = QuantLinear(features, hidden, bits=weight_bits)
        self.feature_affine = HWAffine(hidden)
        self.conv = HWConvLIFBranch(
            conv_channels, weight_bits, act_frac_bits, act_int_bits
        )
        self.fuse_linear = QuantLinear(hidden + conv_channels, 256, bits=weight_bits)
        self.fuse_affine = HWAffine(256)
        self.out = QuantLinear(256, 13, bits=weight_bits)
        self.steps = steps
        self.beta_f = nn.Parameter(torch.tensor(2.2))
        self.beta_o = nn.Parameter(torch.tensor(2.2))
        self.act_frac_bits = act_frac_bits
        self.act_int_bits = act_int_bits

    def _q(self, x: torch.Tensor) -> torch.Tensor:
        return fake_quant_activation(x, self.act_frac_bits, self.act_int_bits)

    def forward(self, features, raw, subject=None):
        f_current = self._q(self.feature_affine(self.feature_linear(features)))
        conv_summary, conv_rates = self.conv(raw)
        mf = torch.zeros_like(f_current)
        mo = f_current.new_zeros(f_current.shape[0], 256)
        logits = f_current.new_zeros(f_current.shape[0], 13)
        rf = f_current.new_zeros(())
        ro = f_current.new_zeros(())
        bf = fake_quant_decay(torch.sigmoid(self.beta_f), self.act_frac_bits)
        bo = fake_quant_decay(torch.sigmoid(self.beta_o), self.act_frac_bits)
        for _ in range(self.steps):
            mf = self._q(bf * mf + f_current)
            sf = spike(mf - 1.0)
            mf = mf - sf
            fused = self._q(
                self.fuse_affine(
                    self.fuse_linear(torch.cat((sf, conv_summary), dim=1))
                )
            )
            mo = self._q(bo * mo + fused)
            so = spike(mo - 1.0)
            mo = mo - so
            logits = logits + self.out(so)
            rf += sf.mean()
            ro += so.mean()
        return logits / self.steps, [rf / self.steps, ro / self.steps, *conv_rates]


def remap_context_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map a trained ClassAdaptiveContextSNN state_dict onto the HW graph."""

    rename = {
        "encoder.0.weight": "enc_linear.weight",
        "encoder.0.bias": "enc_linear.bias",
        "encoder.1.weight": "enc_affine.weight",
        "encoder.1.bias": "enc_affine.bias",
    }
    output: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        output[rename.get(key, key)] = value
    return output


def remap_hybrid_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map a trained HybridSNN state_dict onto the HW graph (drops BN.4 stats
    rename and skips the GELU layer, which has no parameters)."""

    rename = {
        "feature_current.0.weight": "feature_linear.weight",
        "feature_current.0.bias": "feature_linear.bias",
        "feature_current.1.weight": "feature_affine.weight",
        "feature_current.1.bias": "feature_affine.bias",
        "conv.front.0.weight": "conv.conv1.weight",
        "conv.front.0.bias": "conv.conv1.bias",
        "conv.front.1.weight": "conv.bn1.weight",
        "conv.front.1.bias": "conv.bn1.bias",
        "conv.front.1.running_mean": "conv.bn1.running_mean",
        "conv.front.1.running_var": "conv.bn1.running_var",
        "conv.front.1.num_batches_tracked": "conv.bn1.num_batches_tracked",
        "conv.front.3.weight": "conv.conv2.weight",
        "conv.front.3.bias": "conv.conv2.bias",
        "conv.front.4.weight": "conv.bn2.weight",
        "conv.front.4.bias": "conv.bn2.bias",
        "conv.front.4.running_mean": "conv.bn2.running_mean",
        "conv.front.4.running_var": "conv.bn2.running_var",
        "conv.front.4.num_batches_tracked": "conv.bn2.num_batches_tracked",
        "fuse.0.weight": "fuse_linear.weight",
        "fuse.0.bias": "fuse_linear.bias",
        "fuse.1.weight": "fuse_affine.weight",
        "fuse.1.bias": "fuse_affine.bias",
    }
    output: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        output[rename.get(key, key)] = value
    return output
