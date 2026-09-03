from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from prepare import extract_features


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "semg_snn_fpga_reproduction" / "data" / "extracted"
OUT = ROOT / "data"


def majority_nonzero(values):
    values = values[values > 0]
    if not len(values):
        return 0
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[np.argmax(counts)])


def main():
    features, repetitions, subjects, starts_all = [], [], [], []
    offsets = [0]
    for subject in range(1, 11):
        mat = loadmat(SOURCE / f"s{subject}" / f"S{subject}_E1_A1.mat")
        emg = np.asarray(mat["emg"], dtype=np.int8)
        reps = np.asarray(mat["rerepetition"]).reshape(-1).astype(np.int8)
        starts = np.arange(400, len(emg) - 100 + 1, 20, dtype=np.int32)
        subject_features = []
        for begin in range(0, len(starts), 2048):
            chunk_starts = starts[begin : begin + 2048]
            raw = np.stack([emg[start : start + 100] for start in chunk_starts])
            subject_features.append(extract_features(raw))
        features.append(np.concatenate(subject_features).astype(np.float16))
        repetitions.append(np.asarray(
            [majority_nonzero(reps[start : start + 100]) for start in starts],
            dtype=np.int8,
        ))
        subjects.append(np.full(len(starts), subject - 1, dtype=np.int8))
        starts_all.append(starts)
        offsets.append(offsets[-1] + len(starts))
        print(subject, len(starts), flush=True)
    np.save(OUT / "continuous_features.npy", np.concatenate(features))
    np.savez(
        OUT / "continuous_index.npz",
        repetition=np.concatenate(repetitions),
        subject=np.concatenate(subjects),
        start=np.concatenate(starts_all),
        offsets=np.asarray(offsets, dtype=np.int64),
    )
    print("total", offsets[-1], flush=True)


if __name__ == "__main__":
    main()
