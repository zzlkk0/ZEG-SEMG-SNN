"""Three-branch fusion accuracy for the hardware-friendly QAT experts.

Combines the HW-QAT Context and Hybrid branches (hw_model.py) with the
existing (already integer-friendly) Delay-SNN branch, using the same
val-selected simplex weight search + Rest-logit calibration methodology as
evaluate_ensemble.py, so the result is comparable to the FP32 91.0961% /
quantized-proxy 90.7237% numbers documented in
the repository documentation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from train import EMGDataset
from hw_model import HWClassAdaptiveContextSNN, HWHybridSNN


def score(y, probability):
    p = probability.argmax(1)
    return {
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro"),
        "gesture_accuracy": float(np.mean(p[y != 0] == y[y != 0])),
    }


@torch.no_grad()
def predict_context(root: Path, split: str, checkpoint: Path, device):
    dataset = EMGDataset(
        root / "data" / f"{split}.npz",
        root / "data" / "normalization.npz",
        False,
        context=23,
        continuous_context=True,
        stream_context=True,
    )
    model = HWClassAdaptiveContextSNN(dataset.features.shape[1]).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)["model"]
    model.load_state_dict(state)
    model.eval()
    outputs = []
    for f, raw, _, subject in DataLoader(dataset, 512, num_workers=4, pin_memory=True):
        logits, _ = model(f.to(device), raw.to(device), subject.to(device))
        outputs.append(logits.softmax(1).cpu().numpy())
    return np.concatenate(outputs), dataset.y, dataset.subject, dataset.start


@torch.no_grad()
def predict_hybrid(root: Path, split: str, checkpoint: Path, device):
    dataset = EMGDataset(
        root / "data" / f"{split}.npz", root / "data" / "normalization.npz", False, context=1
    )
    model = HWHybridSNN(dataset.features.shape[1]).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)["model"]
    model.load_state_dict(state)
    model.eval()
    outputs = []
    for f, raw, _, subject in DataLoader(dataset, 512, num_workers=4, pin_memory=True):
        logits, _ = model(f.to(device), raw.to(device), subject.to(device))
        outputs.append(logits.softmax(1).cpu().numpy())
    return np.concatenate(outputs), dataset.y, dataset.subject, dataset.start


@torch.no_grad()
def predict_delay(root: Path, split: str, checkpoint: Path, device, temperature: float = 0.03):
    paper_root = root.parent / "semg_snn_fpga_reproduction"
    spec = importlib.util.spec_from_file_location("paper_snn_model", paper_root / "model.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    saved_args = state.get("args", {})
    model = module.PaperSNNWithDelays(
        decay=saved_args.get("decay", 0.9),
        threshold=saved_args.get("threshold", 1.0),
        max_delay=saved_args.get("max_delay", 62),
        initial_delay=saved_args.get("initial_delay", 1.0),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    source = np.load(paper_root / "data" / "processed" / f"{split}.npz")
    x = torch.from_numpy(source["x"].astype(np.float32))
    y = source["y"].astype(np.int64)
    outputs = []
    for (batch,) in DataLoader(TensorDataset(x), 256, num_workers=4, pin_memory=True):
        spikes, _ = model(batch.to(device))
        outputs.append((spikes.mean(1) / temperature).softmax(1).cpu().numpy())
    loop_data = np.load(root / "data" / f"{split}.npz")
    return np.concatenate(outputs), y, loop_data["subject"], loop_data["start"]


def simplex_grid(parts: int, resolution: int):
    def compositions(remaining, count, prefix=()):
        if count == 1:
            yield prefix + (remaining,)
            return
        for value in range(remaining + 1):
            yield from compositions(remaining - value, count - 1, prefix + (value,))

    for values in compositions(resolution, parts):
        yield tuple(value / resolution for value in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--context-checkpoint", type=Path, default=root / "runs/hw_context23_qat_v1/best.pt")
    parser.add_argument("--hybrid-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--delay-checkpoint",
        type=Path,
        default=root.parent / "semg_snn_fpga_reproduction" / "runs" / "delay62_finetune" / "best.pt",
    )
    parser.add_argument("--output", type=Path, default=root / "runs" / "hw_three_branch_fusion_metrics.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probabilities = {"val": [], "test": []}
    labels = {}
    metadata = {}
    for split in probabilities:
        context_p, y, subject, start = predict_context(root, split, args.context_checkpoint, device)
        hybrid_p, y2, _, _ = predict_hybrid(root, split, args.hybrid_checkpoint, device)
        delay_p, y3, _, _ = predict_delay(root, split, args.delay_checkpoint, device)
        assert np.array_equal(y, y2) and np.array_equal(y, y3), f"{split} label misalignment across branches"
        probabilities[split] = [context_p, hybrid_p, delay_p]
        labels[split] = y
        metadata[split] = (subject, start)
        print(f"{split}: context={score(y, context_p)} hybrid={score(y, hybrid_p)} delay={score(y, delay_p)}", flush=True)

    candidates = [
        (a / 10, b / 10, (10 - a - b) / 10) for a in range(11) for b in range(11 - a)
    ]
    rows = []
    for weights in candidates:
        combined = sum(w * p for w, p in zip(weights, probabilities["val"]))
        rows.append({"weights": weights, **score(labels["val"], combined)})
    best = max(rows, key=lambda r: (r["accuracy"], r["macro_f1"]))
    test_probability = sum(w * p for w, p in zip(best["weights"], probabilities["test"]))
    result = {
        "checkpoints": {
            "context": str(args.context_checkpoint),
            "hybrid": str(args.hybrid_checkpoint),
            "delay": str(args.delay_checkpoint),
        },
        "branch_scores": {
            split: {
                "context": score(labels[split], probabilities[split][0]),
                "hybrid": score(labels[split], probabilities[split][1]),
                "delay": score(labels[split], probabilities[split][2]),
            }
            for split in probabilities
        },
        "selected_on_validation": best,
        "test_uncalibrated": score(labels["test"], test_probability),
    }

    val_probability = sum(w * p for w, p in zip(best["weights"], probabilities["val"]))
    val_log_probability = np.log(np.clip(val_probability, 1e-8, 1.0))
    bias_rows = []
    for rest_bias in np.linspace(-1.2, 0.4, 81):
        calibrated = val_log_probability.copy()
        calibrated[:, 0] += rest_bias
        bias_rows.append({"rest_logit_bias": float(rest_bias), **score(labels["val"], calibrated)})
    best_bias = max(bias_rows, key=lambda r: (r["accuracy"], r["macro_f1"]))
    test_calibrated = np.log(np.clip(test_probability, 1e-8, 1.0))
    test_calibrated[:, 0] += best_bias["rest_logit_bias"]
    result["rest_calibration"] = {
        "selected_on_validation": best_bias,
        "test": score(labels["test"], test_calibrated),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
