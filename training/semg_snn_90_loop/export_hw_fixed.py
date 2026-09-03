"""Export HW-QAT Context/Hybrid checkpoints to numpy-only fixed-point bundles.

Reads the trained torch state_dicts (hw_model.py graphs) and writes plain
.npz archives that hw_fixed_reference.py can load and run with numpy only
(no torch import at inference time) -- the same split used by the existing
FixedDelaySNN / fixed_model.py pattern in semg_snn_nexys4ddr_vivado.

Quantization grid (must match hw_model.py / hw_ops.py exactly):
  - Linear/Conv weights: per-output symmetric INT(weight_bits), default INT4.
  - HWAffine (LayerNorm replacement) weight: per-tensor symmetric INT8.
  - Activations / membrane potentials: Q8.8 (act_frac_bits=8, act_int_bits=8).
  - Decay (beta): quantized to k/256.
  - Jaccard reciprocal: real lookup table over union in [0, window_steps],
    frac_bits=12 -- this is the actual LUT hardware would hold, not just a
    fake-quantized float division like the QAT training used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

ACT_FRAC_BITS = 8
ACT_INT_BITS = 8
DECAY_FRAC_BITS = 8
RECIP_FRAC_BITS = 12
CONV_TIME_STEPS = 100


def round_half_even(x: np.ndarray) -> np.ndarray:
    return np.round(x)


def quantize_weight(weight: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    limit = 2 ** (bits - 1) - 1
    reduce_axes = tuple(range(1, weight.ndim))
    amax = np.abs(weight).max(axis=reduce_axes, keepdims=True) if reduce_axes else np.abs(weight).max()
    amax = np.maximum(amax, 1e-8)
    scale = amax / limit
    codes = np.clip(round_half_even(weight / scale), -limit, limit).astype(np.int8)
    return codes, scale.reshape(-1).astype(np.float32)


def quantize_affine(weight: np.ndarray, bits: int = 8) -> tuple[np.ndarray, np.ndarray]:
    limit = 2 ** (bits - 1) - 1
    amax = float(np.maximum(np.abs(weight).max(), 1e-8))
    scale = amax / limit
    codes = np.clip(round_half_even(weight / scale), -limit, limit).astype(np.int8)
    return codes, np.float32(scale)


def quantize_decay(beta: float, frac_bits: int = DECAY_FRAC_BITS) -> float:
    scale = 2.0 ** (-frac_bits)
    limit = 2 ** frac_bits
    q = np.clip(round_half_even(beta / scale), 1, limit)
    return float(q * scale)


def build_reciprocal_table(max_value: int, frac_bits: int = RECIP_FRAC_BITS) -> np.ndarray:
    values = np.arange(0, max_value + 1)
    denom = np.maximum(values, 1)
    scale = 2.0 ** (-frac_bits)
    limit = 2 ** frac_bits
    table = np.clip(round_half_even((1.0 / denom) / scale), 0, limit) * scale
    return table.astype(np.float32)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def export_context(state: dict, weight_bits: int, out_path: Path) -> None:
    arrays: dict[str, np.ndarray] = {}

    def add_linear(prefix: str, weight_key: str, bias_key: str, bits: int) -> None:
        weight = state[weight_key].detach().cpu().numpy()
        bias = state[bias_key].detach().cpu().numpy().astype(np.float32)
        codes, scale = quantize_weight(weight, bits)
        arrays[f"{prefix}_codes"] = codes
        arrays[f"{prefix}_scale"] = scale
        arrays[f"{prefix}_bias"] = bias

    def add_affine(prefix: str, weight_key: str, bias_key: str) -> None:
        weight = state[weight_key].detach().cpu().numpy()
        bias = state[bias_key].detach().cpu().numpy().astype(np.float32)
        codes, scale = quantize_affine(weight)
        arrays[f"{prefix}_codes"] = codes
        arrays[f"{prefix}_scale"] = np.float32(scale)
        arrays[f"{prefix}_bias"] = bias

    add_linear("enc_linear", "enc_linear.weight", "enc_linear.bias", weight_bits)
    add_affine("enc_affine", "enc_affine.weight", "enc_affine.bias")
    add_linear("fc2", "fc2.weight", "fc2.bias", weight_bits)
    add_affine("norm2", "norm2.weight", "norm2.bias")
    add_linear("out", "out.weight", "out.bias", weight_bits)

    beta1 = sigmoid(state["beta1"] + 0.5 * torch.tanh(state["beta1_offset"])).detach().cpu().numpy()
    beta2 = sigmoid(state["beta2"] + 0.5 * torch.tanh(state["beta2_offset"])).detach().cpu().numpy()
    beta1_q = np.array([quantize_decay(float(v)) for v in beta1], dtype=np.float32)
    beta2_q = np.array([quantize_decay(float(v)) for v in beta2], dtype=np.float32)
    arrays["beta1"] = beta1_q
    arrays["beta2"] = beta2_q
    gamma = torch.sigmoid(state["context_gamma_logit"]).detach().cpu().numpy().astype(np.float32)
    arrays["context_gamma"] = gamma

    arrays["act_frac_bits"] = np.int32(ACT_FRAC_BITS)
    arrays["act_int_bits"] = np.int32(ACT_INT_BITS)
    arrays["substeps"] = np.int32(3)
    arrays["windows"] = np.int32(23)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)
    print(f"wrote {out_path} ({sum(a.nbytes for a in arrays.values() if hasattr(a, 'nbytes'))} bytes)")


def export_hybrid(state: dict, weight_bits: int, out_path: Path) -> None:
    arrays: dict[str, np.ndarray] = {}

    def add_linear(prefix: str, weight_key: str, bias_key: str, bits: int) -> None:
        weight = state[weight_key].detach().cpu().numpy()
        bias = state[bias_key].detach().cpu().numpy().astype(np.float32)
        codes, scale = quantize_weight(weight, bits)
        arrays[f"{prefix}_codes"] = codes
        arrays[f"{prefix}_scale"] = scale
        arrays[f"{prefix}_bias"] = bias

    def add_affine(prefix: str, weight_key: str, bias_key: str) -> None:
        weight = state[weight_key].detach().cpu().numpy()
        bias = state[bias_key].detach().cpu().numpy().astype(np.float32)
        codes, scale = quantize_affine(weight)
        arrays[f"{prefix}_codes"] = codes
        arrays[f"{prefix}_scale"] = np.float32(scale)
        arrays[f"{prefix}_bias"] = bias

    def add_bn(prefix: str) -> None:
        weight = state[f"conv.{prefix}.weight"].detach().cpu().numpy().astype(np.float32)
        bias = state[f"conv.{prefix}.bias"].detach().cpu().numpy().astype(np.float32)
        mean = state[f"conv.{prefix}.running_mean"].detach().cpu().numpy().astype(np.float32)
        var = state[f"conv.{prefix}.running_var"].detach().cpu().numpy().astype(np.float32)
        eps = 1e-5
        bn_scale = weight / np.sqrt(var + eps)
        bn_shift = bias - mean * bn_scale
        arrays[f"{prefix}_scale"] = bn_scale
        arrays[f"{prefix}_shift"] = bn_shift

    add_linear("feature_linear", "feature_linear.weight", "feature_linear.bias", weight_bits)
    add_affine("feature_affine", "feature_affine.weight", "feature_affine.bias")
    add_linear("conv1", "conv.conv1.weight", "conv.conv1.bias", weight_bits)
    add_bn("bn1")
    add_linear("conv2", "conv.conv2.weight", "conv.conv2.bias", weight_bits)
    add_bn("bn2")
    add_linear("q", "conv.q.weight", "conv.q.bias", weight_bits)
    add_linear("k", "conv.k.weight", "conv.k.bias", weight_bits)
    add_linear("v", "conv.v.weight", "conv.v.bias", weight_bits)
    add_linear("fuse_linear", "fuse_linear.weight", "fuse_linear.bias", weight_bits)
    add_affine("fuse_affine", "fuse_affine.weight", "fuse_affine.bias")
    add_linear("out", "out.weight", "out.bias", weight_bits)

    arrays["beta_f"] = np.float32(quantize_decay(float(sigmoid(state["beta_f"]))))
    arrays["beta_o"] = np.float32(quantize_decay(float(sigmoid(state["beta_o"]))))
    arrays["conv_beta"] = np.float32(quantize_decay(float(sigmoid(state["conv.beta"]))))

    arrays["reciprocal_table"] = build_reciprocal_table(CONV_TIME_STEPS)
    arrays["act_frac_bits"] = np.int32(ACT_FRAC_BITS)
    arrays["act_int_bits"] = np.int32(ACT_INT_BITS)
    arrays["steps"] = np.int32(12)
    arrays["time_steps"] = np.int32(CONV_TIME_STEPS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)
    print(f"wrote {out_path} ({sum(a.nbytes for a in arrays.values() if hasattr(a, 'nbytes'))} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--context-checkpoint", type=Path, default=root / "runs/hw_context23_qat_v1/best.pt")
    parser.add_argument("--hybrid-checkpoint", type=Path, required=True)
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=root / "weights_hw")
    args = parser.parse_args()

    context_state = torch.load(args.context_checkpoint, map_location="cpu", weights_only=False)["model"]
    export_context(context_state, args.weight_bits, args.out_dir / "hw_context_fixed.npz")

    hybrid_state = torch.load(args.hybrid_checkpoint, map_location="cpu", weights_only=False)["model"]
    export_hybrid(hybrid_state, args.weight_bits, args.out_dir / "hw_hybrid_fixed.npz")


if __name__ == "__main__":
    main()
