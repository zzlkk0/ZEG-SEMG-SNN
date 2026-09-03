from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "semg_snn_fpga_reproduction" / "data" / "extracted"
OUT = ROOT / "data" / "adaptive_delta"
TRAIN_REPS = {1, 2, 4, 6}
REP_SPLIT = {1: "train", 2: "train", 3: "val", 4: "train", 5: "test", 6: "train"}


def majority_nonzero(values):
    values = values[values > 0]
    if not len(values):
        return None
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[np.argmax(counts)])


def delta(signal, threshold):
    time, channels = signal.shape
    pos = np.zeros((time, channels), np.bool_)
    neg = np.zeros_like(pos)
    reference = signal[0].copy()
    for t in range(1, time):
        p = signal[t] > reference + threshold
        n = signal[t] < reference - threshold
        pos[t], neg[t] = p, n
        update = p | n
        reference[update] = signal[t, update]
    return np.concatenate((pos, neg), axis=1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subjects = []
    quantiles = []
    for subject in range(1, 11):
        mat = loadmat(SOURCE / f"s{subject}" / f"S{subject}_E1_A1.mat")
        emg = np.asarray(mat["emg"], dtype=np.float32)
        d1 = np.gradient(emg, axis=0)
        d2 = np.gradient(d1, axis=0)
        labels = np.asarray(mat["restimulus"]).reshape(-1)
        reps = np.asarray(mat["rerepetition"]).reshape(-1)
        mask = np.isin(reps, list(TRAIN_REPS)) & (labels > 0)
        q = np.stack([np.percentile(np.abs(trace[mask]), 90, axis=0)
                      for trace in (emg, d1, d2)])
        quantiles.append(q)
        subjects.append((emg, d1, d2, labels.astype(np.int16), reps.astype(np.int16)))
    quantiles = np.stack(quantiles)
    cohort = np.median(quantiles, axis=0)
    scales = np.clip(quantiles / np.maximum(cohort[None], 1e-3), 0.4, 2.5)
    thresholds = 15.0 * scales
    combined = {s: [] for s in ("train", "val", "test")}
    for subject, (emg, d1, d2, labels, reps) in enumerate(subjects):
        encoded = np.concatenate(
            [delta(trace, thresholds[subject, trace_index])
             for trace_index, trace in enumerate((emg, d1, d2))],
            axis=1,
        )
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
            combined[REP_SPLIT[rep]].append(
                (encoded[start : start + 100], label, subject + 1, start, rep)
            )
    summary = {}
    for split, rows in combined.items():
        arrays = {
            "x": np.stack([r[0] for r in rows]),
            "y": np.asarray([r[1] for r in rows], np.int16),
            "subject": np.asarray([r[2] for r in rows], np.int8),
            "start": np.asarray([r[3] for r in rows], np.int32),
            "repetition": np.asarray([r[4] for r in rows], np.int8),
        }
        np.savez_compressed(OUT / f"{split}.npz", **arrays)
        summary[split] = {"samples": len(rows), "spike_rate": float(arrays["x"].mean())}
        print(split, summary[split], flush=True)
    np.savez(OUT / "thresholds.npz", thresholds=thresholds, scales=scales,
             train_quantiles=quantiles, cohort_quantiles=cohort)
    (OUT / "metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
