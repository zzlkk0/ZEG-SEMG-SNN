from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from model import FeatureSNN, HybridSNN, SubjectFeatureSNN
from train import EMGDataset


WIDTHS = [1, 3, 5, 7, 9, 11, 15, 19, 23, 31, 41, 51]


def make_model(kind: str, features: int):
    if kind == "feature":
        return FeatureSNN(features)
    if kind == "subject":
        return SubjectFeatureSNN(features)
    return HybridSNN(features)


@torch.no_grad()
def infer(model, dataset, loader, device):
    model.eval()
    probabilities = []
    for f, raw, _, subject in loader:
        output, _ = model(f.to(device), raw.to(device), subject.to(device))
        probabilities.append(output.softmax(1).cpu().numpy())
    return np.concatenate(probabilities), dataset.y, dataset.subject


def segments(subject: np.ndarray, starts: np.ndarray):
    begin = 0
    for i in range(1, len(subject)):
        if subject[i] != subject[i - 1] or starts[i] - starts[i - 1] != 20:
            yield begin, i
            begin = i
    yield begin, len(subject)


def smooth(prob: np.ndarray, subject: np.ndarray, starts: np.ndarray,
           width: int, mode: str) -> np.ndarray:
    result = np.zeros_like(prob)
    radius = width // 2
    for begin, end in segments(subject, starts):
        sequence = prob[begin:end]
        cumulative = np.vstack((np.zeros((1, prob.shape[1])), np.cumsum(sequence, axis=0)))
        for local in range(len(sequence)):
            if mode == "centered":
                left, right = max(0, local - radius), min(len(sequence), local + radius + 1)
            else:
                left, right = max(0, local - width + 1), local + 1
            result[begin + local] = (cumulative[right] - cumulative[left]) / (right - left)
    return result


def metrics(y, probability):
    prediction = probability.argmax(1)
    return {
        "accuracy": accuracy_score(y, prediction),
        "macro_f1": f1_score(y, prediction, average="macro"),
        "gesture_accuracy": float(np.mean(prediction[y != 0] == y[y != 0])),
    }


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["feature", "subject", "hybrid"], required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = root / "data"
    datasets = {
        split: EMGDataset(data / f"{split}.npz", data / "normalization.npz", False)
        for split in ("val", "test")
    }
    model = make_model(args.model, datasets["val"].features.shape[1]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    inferred = {}
    for split, dataset in datasets.items():
        loader = DataLoader(dataset, args.batch_size, num_workers=4, pin_memory=True)
        probability, y, subject = infer(model, dataset, loader, device)
        starts = np.load(data / f"{split}.npz")["start"]
        inferred[split] = (probability, y, subject, starts)

    selection = {}
    for mode in ("causal", "centered"):
        candidates = []
        pv, yv, sv, tv = inferred["val"]
        for width in WIDTHS:
            row = {"width": width, **metrics(yv, smooth(pv, sv, tv, width, mode))}
            candidates.append(row)
        best = max(candidates, key=lambda r: (r["accuracy"], r["macro_f1"]))
        pt, yt, st, tt = inferred["test"]
        test = metrics(yt, smooth(pt, st, tt, best["width"], mode))
        selection[mode] = {
            "selected_on_validation": best,
            "test": test,
            "future_context_seconds": 0.0 if mode == "causal" else (best["width"] // 2) * 0.1,
            "all_validation_candidates": candidates,
        }
    selection["raw"] = {
        "validation": metrics(inferred["val"][1], inferred["val"][0]),
        "test": metrics(inferred["test"][1], inferred["test"][0]),
    }
    output = args.checkpoint.parent / "sequence_metrics.json"
    output.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps({k: ({q: v[q] for q in v if q != "all_validation_candidates"}
                          if k != "raw" else v) for k, v in selection.items()}, indent=2))


if __name__ == "__main__":
    main()
