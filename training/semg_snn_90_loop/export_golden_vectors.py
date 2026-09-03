"""Golden test vectors for the HW-QAT Context/Hybrid numpy fixed-point graphs.

Picks one strict-test-set window per class (13) plus a few extra repeats to
reach ~19 samples, runs hw_fixed_reference.py end to end, and records inputs,
key intermediate checkpoints and final logits/argmax so an HLS/RTL sim can be
cross-checked against this numpy reference.  This is a starting artifact: the
reference is deterministic, so a finer per-timestep trace can be regenerated
from the same weights_hw/*.npz on demand.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hw_fixed_reference import (
    HWFixedContext,
    HWFixedHybrid,
    affine,
    linear,
    spike,
)
from train import EMGDataset

ROOT = Path(__file__).resolve().parent
LABELS = (
    "Rest", "Index flexion", "Index extension", "Middle flexion", "Middle extension",
    "Ring flexion", "Ring extension", "Little flexion", "Little extension",
    "Thumb adduction", "Thumb abduction", "Thumb flexion", "Thumb extension",
)


def traced_context(model: HWFixedContext, features: np.ndarray) -> dict:
    w = model.w
    batch, windows, _ = features.shape
    hidden1 = w["enc_linear_bias"].shape[0]
    hidden2 = w["fc2_bias"].shape[0]
    m1 = np.zeros((batch, hidden1), dtype=np.float32)
    m2 = np.zeros((batch, hidden2), dtype=np.float32)
    output = np.zeros((batch, 13), dtype=np.float32)
    b1, b2 = w["beta1"][None], w["beta2"][None]
    gamma = w["context_gamma"]
    lags = np.arange(windows - 1, -1, -1)
    weights = gamma[None, :] ** lags[:, None]
    spike_totals = {"s1": np.zeros(batch), "s2": np.zeros(batch)}
    first_window_current1 = None
    for window in range(windows):
        current1 = model._q(
            affine(
                linear(features[:, window], w["enc_linear_codes"], w["enc_linear_scale"], w["enc_linear_bias"]),
                w["enc_affine_codes"], w["enc_affine_scale"], w["enc_affine_bias"],
            )
        )
        if window == 0:
            first_window_current1 = current1.copy()
        for _ in range(model.substeps):
            m1 = model._q(b1 * m1 + current1)
            s1 = spike(m1 - 1.0)
            m1 = m1 - s1
            spike_totals["s1"] += s1.sum(axis=1)
            current2 = model._q(
                affine(
                    linear(s1, w["fc2_codes"], w["fc2_scale"], w["fc2_bias"]),
                    w["norm2_codes"], w["norm2_scale"], w["norm2_bias"],
                )
            )
            m2 = model._q(b2 * m2 + current2)
            s2 = spike(m2 - 1.0)
            m2 = m2 - s2
            spike_totals["s2"] += s2.sum(axis=1)
            output = output + weights[window] * linear(s2, w["out_codes"], w["out_scale"], w["out_bias"])
    normalization = model.substeps * weights.sum(axis=0)
    logits = output / normalization
    return {
        "first_window_encoder_out": first_window_current1,
        "final_membrane1": m1,
        "final_membrane2": m2,
        "spike_count_layer1": spike_totals["s1"],
        "spike_count_layer2": spike_totals["s2"],
        "logits": logits,
        "argmax": logits.argmax(axis=1),
    }


def traced_hybrid(model: HWFixedHybrid, features: np.ndarray, raw: np.ndarray) -> dict:
    w = model.w
    f_current = model._q(
        affine(
            linear(features, w["feature_linear_codes"], w["feature_linear_scale"], w["feature_linear_bias"]),
            w["feature_affine_codes"], w["feature_affine_scale"], w["feature_affine_bias"],
        )
    )
    conv_summary = model._conv_branch(raw)
    batch = features.shape[0]
    mf = np.zeros_like(f_current)
    mo = np.zeros((batch, w["fuse_linear_bias"].shape[0]), dtype=np.float32)
    logits = np.zeros((batch, 13), dtype=np.float32)
    bf, bo = float(w["beta_f"]), float(w["beta_o"])
    sf_total = np.zeros(batch)
    so_total = np.zeros(batch)
    for _ in range(model.steps):
        mf = model._q(bf * mf + f_current)
        sf = spike(mf - 1.0)
        mf = mf - sf
        sf_total += sf.sum(axis=1)
        fused = model._q(
            affine(
                linear(
                    np.concatenate((sf, conv_summary), axis=1),
                    w["fuse_linear_codes"], w["fuse_linear_scale"], w["fuse_linear_bias"],
                ),
                w["fuse_affine_codes"], w["fuse_affine_scale"], w["fuse_affine_bias"],
            )
        )
        mo = model._q(bo * mo + fused)
        so = spike(mo - 1.0)
        mo = mo - so
        so_total += so.sum(axis=1)
        logits = logits + linear(so, w["out_codes"], w["out_scale"], w["out_bias"])
    logits = logits / model.steps
    return {
        "feature_current": f_current,
        "conv_branch_summary": conv_summary,
        "final_membrane_f": mf,
        "final_membrane_o": mo,
        "spike_count_f": sf_total,
        "spike_count_o": so_total,
        "logits": logits,
        "argmax": logits.argmax(axis=1),
    }


def main() -> None:
    data = ROOT / "data"
    context_ds = EMGDataset(
        data / "test.npz", data / "normalization.npz", False,
        context=23, continuous_context=True, stream_context=True,
    )
    hybrid_ds = EMGDataset(data / "test.npz", data / "normalization.npz", False, context=1)

    labels = context_ds.y
    picked: list[int] = []
    for class_id in range(13):
        indices = np.flatnonzero(labels == class_id)
        if len(indices):
            picked.append(int(indices[0]))
    extra = np.flatnonzero(labels != 0)[:6]
    picked.extend(int(i) for i in extra if i not in picked)
    picked = picked[:19]

    context_model = HWFixedContext(ROOT / "weights_hw" / "hw_context_fixed.npz")
    hybrid_model = HWFixedHybrid(ROOT / "weights_hw" / "hw_hybrid_fixed.npz")

    context_features = np.stack([context_ds[i][0].numpy() for i in picked]).astype(np.float32)
    hybrid_features = np.stack([hybrid_ds[i][0].numpy() for i in picked]).astype(np.float32)
    hybrid_raw = np.stack([hybrid_ds[i][1].numpy() for i in picked]).astype(np.float32)
    truth = np.asarray([int(labels[i]) for i in picked])

    context_trace = traced_context(context_model, context_features)
    hybrid_trace = traced_hybrid(hybrid_model, hybrid_features, hybrid_raw)

    out_path = ROOT / "weights_hw" / "golden_vectors.npz"
    np.savez(
        out_path,
        sample_index=np.asarray(picked),
        truth=truth,
        context_features=context_features,
        hybrid_features=hybrid_features,
        hybrid_raw=hybrid_raw,
        **{f"context_{k}": v for k, v in context_trace.items()},
        **{f"hybrid_{k}": v for k, v in hybrid_trace.items()},
    )

    summary = []
    for row, idx in enumerate(picked):
        summary.append({
            "sample_index": idx,
            "truth_id": int(truth[row]),
            "truth_label": LABELS[int(truth[row])],
            "context_argmax": int(context_trace["argmax"][row]),
            "hybrid_argmax": int(hybrid_trace["argmax"][row]),
            "context_correct": bool(context_trace["argmax"][row] == truth[row]),
            "hybrid_correct": bool(hybrid_trace["argmax"][row] == truth[row]),
        })
    summary_path = ROOT / "weights_hw" / "golden_vectors_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"wrote {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
