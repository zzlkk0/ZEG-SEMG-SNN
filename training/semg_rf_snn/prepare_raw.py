from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "semg_snn_fpga_reproduction"
OUT = ROOT / "data"
REP_SPLITS = {1: "train", 2: "train", 3: "val", 4: "train", 5: "test", 6: "train"}


def majority_nonzero(values: np.ndarray) -> int | None:
    values = values[values > 0]
    if len(values) == 0:
        return None
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[np.argmax(counts)])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {}
    assembled = {split: {"x": [], "y": [], "subject": []} for split in ("train", "val", "test")}
    for subject in range(1, 11):
        mat = loadmat(
            SOURCE / "data" / "extracted" / f"s{subject}" / f"S{subject}_E1_A1.mat"
        )
        emg = np.asarray(mat["emg"], dtype=np.float32)
        labels = np.asarray(mat["restimulus"]).reshape(-1).astype(np.int16)
        reps = np.asarray(mat["rerepetition"]).reshape(-1).astype(np.int16)
        for start in range(400, len(emg) - 100 + 1, 20):
            window_labels = labels[start : start + 100]
            if np.all(window_labels == 0):
                label = 0
            else:
                label = majority_nonzero(window_labels)
                if label is None or np.count_nonzero(window_labels == label) / 100 < 0.8:
                    continue
            repetition = majority_nonzero(reps[start : start + 100])
            if repetition is None:
                continue
            split = REP_SPLITS[repetition]
            center = start + 24
            assembled[split]["x"].append(emg[center : center + 52])
            assembled[split]["y"].append(label)
            assembled[split]["subject"].append(subject)

    train_values = []
    for split, values in assembled.items():
        x = np.stack(values["x"]).astype(np.float32)
        y = np.asarray(values["y"], dtype=np.int64)
        subject_ids = np.asarray(values["subject"], dtype=np.int8)
        if split == "train":
            train_values.append(x.reshape(-1, 16))
        np.savez_compressed(OUT / f"{split}_raw.npz", x=x, y=y, subject=subject_ids)
        metadata[split] = {"samples": len(y), "shape": list(x.shape)}
        print(split, x.shape, flush=True)

    train = np.concatenate(train_values)
    # Robust per-channel normalization; resistant to occasional EMG outliers.
    median = np.median(train, axis=0)
    scale = np.percentile(np.abs(train - median), 95, axis=0)
    scale = np.maximum(scale, 1.0)
    np.savez(OUT / "normalization.npz", median=median, scale=scale)
    metadata["normalization"] = {"center": "median", "scale": "95th absolute deviation"}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
