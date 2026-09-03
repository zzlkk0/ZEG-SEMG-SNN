from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from evaluate_ensemble import score
from model import ClassAdaptiveContextSNN


class FullStreamDataset(Dataset):
    def __init__(self, root: Path, context: int):
        self.features = np.load(root / "continuous_features.npy", mmap_mode="r")
        index = np.load(root / "continuous_index.npz")
        self.subject = index["subject"].astype(np.int64)
        self.offsets = index["offsets"].astype(np.int64)
        normalization = np.load(root / "normalization.npz")
        self.mean = normalization["feature_mean"]
        self.std = normalization["feature_std"]
        self.context = context

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        subject = int(self.subject[item])
        first = int(self.offsets[subject])
        start = max(first, item - self.context + 1)
        indices = np.arange(start, item + 1)
        if len(indices) < self.context:
            indices = np.pad(
                indices, (self.context - len(indices), 0), constant_values=first
            )
        x = (
            self.features[indices].astype(np.float32) - self.mean[subject]
        ) / self.std[subject]
        return torch.from_numpy(np.clip(x, -8, 8))


def causal_average(probability, subjects, width):
    output = np.empty_like(probability)
    for subject in range(10):
        indices = np.flatnonzero(subjects == subject)
        p = probability[indices]
        cumulative = np.concatenate(
            (np.zeros((1, p.shape[1]), dtype=np.float64), np.cumsum(p, axis=0)),
            axis=0,
        )
        starts = np.maximum(np.arange(len(p)) + 1 - width, 0)
        sums = cumulative[np.arange(len(p)) + 1] - cumulative[starts]
        output[indices] = sums / (np.arange(len(p)) + 1 - starts)[:, None]
    return output


def target_indices(data_root, split):
    target = np.load(data_root / f"{split}.npz")
    index = np.load(data_root / "continuous_index.npz")
    offsets = index["offsets"]
    positions = offsets[target["subject"]] + (target["start"] - 400) // 20
    return positions.astype(np.int64), target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=23)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/full_stream_metrics.json")
    )
    parser.add_argument(
        "--probabilities",
        type=Path,
        default=Path("runs/full_stream_probabilities.npz"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data_root = root / "data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FullStreamDataset(data_root, args.context)
    model = ClassAdaptiveContextSNN(dataset.features.shape[1]).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    pieces = []
    with torch.no_grad():
        for features in DataLoader(
            dataset, batch_size=512, num_workers=4, pin_memory=True
        ):
            logits, _ = model(features.to(device))
            pieces.append(logits.softmax(1).cpu().numpy())
    full_probability = np.concatenate(pieces)
    val_positions, val = target_indices(data_root, "val")
    test_positions, test = target_indices(data_root, "test")
    widths = (1, 3, 5, 7, 9, 11, 15, 21, 31)
    rows, smoothed = [], {}
    for width in widths:
        p = causal_average(full_probability, dataset.subject, width)
        smoothed[width] = p
        rows.append({
            "width": width,
            **score(val["y"], p[val_positions]),
        })
    selected = max(rows, key=lambda row: (row["accuracy"], row["macro_f1"]))
    chosen = smoothed[selected["width"]]
    result = {
        "protocol": (
            "Inference and causal averaging over every continuous 100-ms-hop "
            "window; subject boundary is the only reset."
        ),
        "checkpoint": str(args.checkpoint),
        "context": args.context,
        "selected_on_validation": selected,
        "test": score(test["y"], chosen[test_positions]),
        "candidates": rows,
    }
    output = args.output if args.output.is_absolute() else root / args.output
    probability_output = (
        args.probabilities if args.probabilities.is_absolute()
        else root / args.probabilities
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    probability_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(
        probability_output,
        val=chosen[val_positions],
        test=chosen[test_positions],
        width=selected["width"],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
