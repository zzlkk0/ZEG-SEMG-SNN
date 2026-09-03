from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "semg_snn_fpga_reproduction"
REPS = {1: "train", 2: "train", 3: "val", 4: "train", 5: "test", 6: "train"}


def majority_nonzero(a: np.ndarray) -> int | None:
    a = a[a > 0]
    if not len(a):
        return None
    u, c = np.unique(a, return_counts=True)
    return int(u[np.argmax(c)])


def extract_features(x: np.ndarray) -> np.ndarray:
    """Time-domain and spectral EMG features, returned as float32."""
    x = x.astype(np.float32)
    dx = np.diff(x, axis=1)
    eps = 1e-5
    feats = [
        np.mean(np.abs(x), axis=1),
        np.sqrt(np.mean(x * x, axis=1) + eps),
        np.std(x, axis=1),
        np.mean(np.abs(dx), axis=1),
        np.log(np.var(x, axis=1) + 1.0),
        np.mean(np.abs(dx) > 4.0, axis=1),
        np.max(x, axis=1) - np.min(x, axis=1),
        np.mean(np.abs(x - np.mean(x, axis=1, keepdims=True)), axis=1),
    ]
    # Thresholded zero crossings and slope sign changes suppress ADC noise.
    zc = ((x[:, 1:] * x[:, :-1] < 0) & (np.abs(dx) > 3.0)).mean(axis=1)
    d1, d2 = dx[:, :-1], dx[:, 1:]
    ssc = ((d1 * d2 < 0) & (np.abs(d1 - d2) > 3.0)).mean(axis=1)
    feats.extend((zc, ssc))
    spectrum = np.abs(np.fft.rfft(x, axis=1)) ** 2
    freqs = np.fft.rfftfreq(x.shape[1], 1 / 200.0)
    total = spectrum[:, 1:].sum(axis=1) + eps
    for low, high in ((0, 20), (20, 40), (40, 60), (60, 80), (80, 101)):
        mask = (freqs >= low) & (freqs < high)
        feats.append(np.log1p(spectrum[:, mask].sum(axis=1)))
        feats.append(spectrum[:, mask].sum(axis=1) / total)
    centroid = (spectrum * freqs[None, :, None]).sum(axis=1) / (spectrum.sum(axis=1) + eps)
    feats.append(centroid / 100.0)
    return np.concatenate(feats, axis=1).astype(np.float32)


def main() -> None:
    out = ROOT / "data"
    out.mkdir(parents=True, exist_ok=True)
    rows = {s: {"raw": [], "y": [], "subject": [], "start": []} for s in REPS.values()}
    # Dict comprehension above duplicates values but intentionally gives three keys.
    for file_subject in range(1, 11):
        mat = loadmat(SOURCE / "data" / "extracted" / f"s{file_subject}" / f"S{file_subject}_E1_A1.mat")
        emg = np.asarray(mat["emg"], dtype=np.int8)
        labels = np.asarray(mat["restimulus"]).reshape(-1).astype(np.int16)
        reps = np.asarray(mat["rerepetition"]).reshape(-1).astype(np.int16)
        for start in range(400, len(emg) - 100 + 1, 20):
            lw = labels[start : start + 100]
            if np.all(lw == 0):
                label = 0
            else:
                label = majority_nonzero(lw)
                if label is None or np.count_nonzero(lw == label) / 100 < 0.8:
                    continue
            rep = majority_nonzero(reps[start : start + 100])
            if rep is None:
                continue
            split = REPS[rep]
            rows[split]["raw"].append(emg[start : start + 100])
            rows[split]["y"].append(label)
            rows[split]["subject"].append(file_subject - 1)
            rows[split]["start"].append(start)

    summary = {}
    all_arrays = {}
    for split, values in rows.items():
        raw = np.stack(values["raw"])
        arrays = {
            "raw": raw,
            "features": extract_features(raw),
            "y": np.asarray(values["y"], dtype=np.int8),
            "subject": np.asarray(values["subject"], dtype=np.int8),
            "start": np.asarray(values["start"], dtype=np.int32),
        }
        all_arrays[split] = arrays
        summary[split] = {"samples": len(arrays["y"]), "raw": list(raw.shape),
                          "features": list(arrays["features"].shape)}
        print(split, summary[split], flush=True)

    train = all_arrays["train"]
    # Subject-specific statistics are fit exclusively on training repetitions.
    feature_mean = np.zeros((10, train["features"].shape[1]), np.float32)
    feature_std = np.ones_like(feature_mean)
    raw_mean = np.zeros((10, 16), np.float32)
    raw_std = np.ones_like(raw_mean)
    for subject in range(10):
        mask = train["subject"] == subject
        feature_mean[subject] = train["features"][mask].mean(axis=0)
        feature_std[subject] = train["features"][mask].std(axis=0) + 1e-4
        raw_float = train["raw"][mask].astype(np.float32)
        raw_mean[subject] = raw_float.mean(axis=(0, 1))
        raw_std[subject] = raw_float.std(axis=(0, 1)) + 1e-4
    np.savez(out / "normalization.npz", feature_mean=feature_mean, feature_std=feature_std,
             raw_mean=raw_mean, raw_std=raw_std)
    for split, arrays in all_arrays.items():
        np.savez_compressed(out / f"{split}.npz", **arrays)
    (out / "metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
