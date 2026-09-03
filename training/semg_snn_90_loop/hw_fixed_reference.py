"""numpy-only (no torch) fixed-point inference for the HW-QAT Context/Hybrid
branches.

Loads the .npz bundles written by export_hw_fixed.py and reproduces the
hw_model.py forward graph exactly: INT(weight_bits) per-output weights,
Q8.8 fixed-point activations/membrane potentials, k/256 decay, ReLU6,
BatchNorm folded to a post-conv affine, and a genuine lookup table for the
Jaccard-attention reciprocal (no runtime division).  Softmax/argmax fusion
stays off this module, matching the rest of the deployed pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def round_half_even(x: np.ndarray) -> np.ndarray:
    return np.round(x)


def fake_quant_activation(x: np.ndarray, frac_bits: int, int_bits: int) -> np.ndarray:
    scale = 2.0 ** (-frac_bits)
    limit = 2 ** (int_bits + frac_bits - 1) - 1
    q = np.clip(round_half_even(x / scale), -limit, limit)
    return (q * scale).astype(np.float32)


def spike(x: np.ndarray) -> np.ndarray:
    return (x >= 0).astype(np.float32)


def relu6(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 6.0)


def linear(x: np.ndarray, codes: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    weight = codes.astype(np.float32) * scale[:, None]
    return x @ weight.T + bias


def affine(x: np.ndarray, codes: np.ndarray, scale: np.float32, bias: np.ndarray) -> np.ndarray:
    weight = codes.astype(np.float32) * scale
    return x * weight + bias


def conv1d(x: np.ndarray, codes: np.ndarray, scale: np.ndarray, bias: np.ndarray, padding: int) -> np.ndarray:
    """x: [batch,time,in]; codes: [out,in,kernel] -> returns [batch,time,out]."""

    weight = codes.astype(np.float32) * scale[:, None, None]
    batch, time, _ = x.shape
    out_channels, _, kernel = weight.shape
    x_padded = np.pad(x, ((0, 0), (padding, padding), (0, 0)))
    out = np.zeros((batch, time, out_channels), dtype=np.float32)
    for k in range(kernel):
        segment = x_padded[:, k : k + time, :]
        out += segment @ weight[:, :, k].T
    out += bias
    return out


class HWFixedContext:
    def __init__(self, path: str | Path) -> None:
        with np.load(path) as archive:
            self.w = {k: archive[k] for k in archive.files}
        self.act_frac_bits = int(self.w["act_frac_bits"])
        self.act_int_bits = int(self.w["act_int_bits"])
        self.substeps = int(self.w["substeps"])

    def _q(self, x: np.ndarray) -> np.ndarray:
        return fake_quant_activation(x, self.act_frac_bits, self.act_int_bits)

    def infer(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """features: [batch,windows,336] -> (logits [batch,13], argmax [batch])."""

        w = self.w
        batch, windows, _ = features.shape
        hidden1 = w["enc_linear_bias"].shape[0]
        hidden2 = w["fc2_bias"].shape[0]
        m1 = np.zeros((batch, hidden1), dtype=np.float32)
        m2 = np.zeros((batch, hidden2), dtype=np.float32)
        output = np.zeros((batch, 13), dtype=np.float32)
        b1 = w["beta1"][None]
        b2 = w["beta2"][None]
        gamma = w["context_gamma"]
        lags = np.arange(windows - 1, -1, -1)
        weights = gamma[None, :] ** lags[:, None]
        for window in range(windows):
            current1 = self._q(
                affine(
                    linear(features[:, window], w["enc_linear_codes"], w["enc_linear_scale"], w["enc_linear_bias"]),
                    w["enc_affine_codes"], w["enc_affine_scale"], w["enc_affine_bias"],
                )
            )
            for _ in range(self.substeps):
                m1 = self._q(b1 * m1 + current1)
                s1 = spike(m1 - 1.0)
                m1 = m1 - s1
                current2 = self._q(
                    affine(
                        linear(s1, w["fc2_codes"], w["fc2_scale"], w["fc2_bias"]),
                        w["norm2_codes"], w["norm2_scale"], w["norm2_bias"],
                    )
                )
                m2 = self._q(b2 * m2 + current2)
                s2 = spike(m2 - 1.0)
                m2 = m2 - s2
                output = output + weights[window] * linear(s2, w["out_codes"], w["out_scale"], w["out_bias"])
        normalization = self.substeps * weights.sum(axis=0)
        logits = output / normalization
        return logits, logits.argmax(axis=1)


class HWFixedHybrid:
    def __init__(self, path: str | Path) -> None:
        with np.load(path) as archive:
            self.w = {k: archive[k] for k in archive.files}
        self.act_frac_bits = int(self.w["act_frac_bits"])
        self.act_int_bits = int(self.w["act_int_bits"])
        self.steps = int(self.w["steps"])
        self.time_steps = int(self.w["time_steps"])
        self.reciprocal_table = self.w["reciprocal_table"]

    def _q(self, x: np.ndarray) -> np.ndarray:
        return fake_quant_activation(x, self.act_frac_bits, self.act_int_bits)

    def _conv_branch(self, raw: np.ndarray) -> np.ndarray:
        w = self.w
        x = conv1d(raw, w["conv1_codes"], w["conv1_scale"], w["conv1_bias"], padding=3)
        x = relu6(x * w["bn1_scale"] + w["bn1_shift"])
        current = conv1d(x, w["conv2_codes"], w["conv2_scale"], w["conv2_bias"], padding=2)
        current = current * w["bn2_scale"] + w["bn2_shift"]

        batch, time, channels = current.shape
        membrane = np.zeros((batch, channels), dtype=np.float32)
        beta = float(w["conv_beta"])
        events = np.zeros((batch, time, channels), dtype=np.float32)
        for t in range(time):
            membrane = self._q(beta * membrane + current[:, t])
            e = spike(membrane - 1.0)
            membrane = membrane - e
            events[:, t] = e

        # 1x1 convs are per-timestep linear layers; apply directly.
        query = spike(np.einsum("btc,oc->bto", events, w["q_codes"][:, :, 0].astype(np.float32) * w["q_scale"][:, None]) + w["q_bias"] - 0.5)
        key = spike(np.einsum("btc,oc->bto", events, w["k_codes"][:, :, 0].astype(np.float32) * w["k_scale"][:, None]) + w["k_bias"] - 0.5)
        value = spike(np.einsum("btc,oc->bto", events, w["v_codes"][:, :, 0].astype(np.float32) * w["v_scale"][:, None]) + w["v_bias"] - 0.5)

        intersection = np.minimum(query, key).sum(axis=1)
        union = np.maximum(query, key).sum(axis=1)
        union_index = np.clip(union, 0, self.time_steps).astype(np.int64)
        reciprocal = self.reciprocal_table[union_index]
        attention = intersection * reciprocal
        attended = value * attention[:, None, :] + events
        return attended.mean(axis=1)

    def infer(self, features: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """features: [batch,336]; raw: [batch,time,16] -> (logits, argmax)."""

        w = self.w
        f_current = self._q(
            affine(
                linear(features, w["feature_linear_codes"], w["feature_linear_scale"], w["feature_linear_bias"]),
                w["feature_affine_codes"], w["feature_affine_scale"], w["feature_affine_bias"],
            )
        )
        conv_summary = self._conv_branch(raw)
        batch = features.shape[0]
        mf = np.zeros_like(f_current)
        mo = np.zeros((batch, w["fuse_linear_bias"].shape[0]), dtype=np.float32)
        logits = np.zeros((batch, 13), dtype=np.float32)
        bf = float(w["beta_f"])
        bo = float(w["beta_o"])
        for _ in range(self.steps):
            mf = self._q(bf * mf + f_current)
            sf = spike(mf - 1.0)
            mf = mf - sf
            fused = self._q(
                affine(
                    linear(
                        np.concatenate((sf, conv_summary), axis=1),
                        w["fuse_linear_codes"], w["fuse_linear_scale"], w["fuse_linear_bias"],
                    ),
                    w["fuse_affine_codes"], w["fuse_affine_scale"], w["fuse_affine_bias"],
                )
            )
            mo = self._q(bo * mo + fused)
            so = spike(mo - 1.0)
            mo = mo - so
            logits = logits + linear(so, w["out_codes"], w["out_scale"], w["out_bias"])
        logits = logits / self.steps
        return logits, logits.argmax(axis=1)
