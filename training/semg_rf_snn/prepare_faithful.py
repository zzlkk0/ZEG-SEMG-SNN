from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "semg_snn_fpga_reproduction" / "data" / "extracted"
PROTOCOLS = {
    "paper": {"train": {1, 3, 4, 6}, "val": {2, 5}, "test": set()},
    "strict": {"train": {1, 2, 4, 6}, "val": {3}, "test": {5}},
}


def majority_nonzero(values: np.ndarray) -> int | None:
    values = values[values > 0]
    if not len(values):
        return None
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[np.argmax(counts)])


def split_for(rep: int, protocol: str) -> str | None:
    for split, repetitions in PROTOCOLS[protocol].items():
        if rep in repetitions:
            return split
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=PROTOCOLS, required=True)
    parser.add_argument("--purity", type=float, default=0.8)
    args = parser.parse_args()
    destination = ROOT / "data" / f"faithful_{args.protocol}"
    destination.mkdir(parents=True, exist_ok=True)
    rows = {s: {"x": [], "y": [], "subject": [], "start": []} for s in ("train", "val", "test")}

    for subject in range(1, 11):
        mat = loadmat(SOURCE / f"s{subject}" / f"S{subject}_E1_A1.mat")
        emg = np.asarray(mat["emg"], dtype=np.float32)
        labels = np.asarray(mat["restimulus"]).reshape(-1).astype(np.int16)
        reps = np.asarray(mat["rerepetition"]).reshape(-1).astype(np.int16)
        # 52 samples = 260 ms at 200 Hz; a 10-sample stride is ~80.8% overlap.
        for start in range(0, len(emg) - 52 + 1, 10):
            window_labels = labels[start : start + 52]
            if np.all(window_labels == 0):
                label = 0
            else:
                label = majority_nonzero(window_labels)
                if label is None or np.count_nonzero(window_labels == label) / 52 < args.purity:
                    continue
            rep = majority_nonzero(reps[start : start + 52])
            if rep is None:
                continue
            split = split_for(rep, args.protocol)
            if split is None:
                continue
            rows[split]["x"].append(emg[start : start + 52])
            rows[split]["y"].append(label)
            rows[split]["subject"].append(subject)
            rows[split]["start"].append(start)

    summary = {}
    for split, values in rows.items():
        if not values["x"]:
            continue
        arrays = {
            "x": np.stack(values["x"]).astype(np.int8),
            "y": np.asarray(values["y"], dtype=np.int8),
            "subject": np.asarray(values["subject"], dtype=np.int8),
            "start": np.asarray(values["start"], dtype=np.int32),
        }
        np.savez_compressed(destination / f"{split}.npz", **arrays)
        summary[split] = {
            "samples": len(arrays["y"]),
            "shape": list(arrays["x"].shape),
            "classes": np.bincount(arrays["y"], minlength=13).tolist(),
        }
        print(split, summary[split], flush=True)
    metadata = {
        "protocol": args.protocol,
        "repetitions": {k: sorted(v) for k, v in PROTOCOLS[args.protocol].items()},
        "window": 52,
        "stride": 10,
        "overlap": 1 - 10 / 52,
        "purity": args.purity,
        "splits": summary,
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
