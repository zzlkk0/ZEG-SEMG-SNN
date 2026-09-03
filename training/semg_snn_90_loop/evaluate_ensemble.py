from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from model import (
    AdaptiveContextSNN,
    ClassAdaptiveContextSNN,
    ContextHybridSNN,
    ContextSNN,
    FeatureSNN,
    HybridSNN,
)
from train import EMGDataset
from evaluate_sequence import smooth


def score(y, probability):
    p = probability.argmax(1)
    return {
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro"),
        "gesture_accuracy": float(np.mean(p[y != 0] == y[y != 0])),
    }


def simplex_grid(parts, resolution):
    """Yield non-negative simplex weights on an integer grid."""
    if parts == 1:
        yield (1.0,)
        return

    def compositions(remaining, count, prefix=()):
        if count == 1:
            yield prefix + (remaining,)
            return
        for value in range(remaining + 1):
            yield from compositions(remaining - value, count - 1, prefix + (value,))

    for values in compositions(resolution, parts):
        yield tuple(value / resolution for value in values)


@torch.no_grad()
def predict(root, split, kind, context, checkpoint, device):
    if kind == "delta":
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
        source = np.load(paper_root / "data" / "processed" / f"{split}.npz")
        x = torch.from_numpy(source["x"].astype(np.float32))
        y = source["y"].astype(np.int64)
        outputs = []
        model.eval()
        for (batch,) in DataLoader(TensorDataset(x), 256, num_workers=4, pin_memory=True):
            spikes, _ = model(batch.to(device))
            temperature = context / 100.0 if context > 0 else 0.05
            outputs.append((spikes.mean(1) / temperature).softmax(1).cpu().numpy())
        loop_data = np.load(root / "data" / f"{split}.npz")
        return np.concatenate(outputs), y, loop_data["subject"], loop_data["start"]
    continuous = kind in (
        "context_continuous", "context_adaptive_continuous",
        "context_class_adaptive_continuous",
        "context_hybrid_continuous",
        "context_stream", "context_adaptive_stream",
        "context_class_adaptive_stream", "context_hybrid_stream",
    )
    stream = kind.endswith("_stream")
    dataset = EMGDataset(
        root / "data" / f"{split}.npz",
        root / "data" / "normalization.npz",
        False,
        context=context,
        continuous_context=continuous,
        stream_context=stream,
    )
    features = dataset.features.shape[1]
    if kind in ("context", "context_continuous", "context_stream"):
        model = ContextSNN(features)
    elif kind in (
        "context_adaptive", "context_adaptive_continuous",
        "context_adaptive_stream",
    ):
        model = AdaptiveContextSNN(features)
    elif kind in (
        "context_class_adaptive", "context_class_adaptive_continuous",
        "context_class_adaptive_stream",
    ):
        model = ClassAdaptiveContextSNN(features)
    elif kind in (
        "context_hybrid", "context_hybrid_continuous", "context_hybrid_stream"
    ):
        model = ContextHybridSNN(features)
    elif kind == "hybrid":
        model = HybridSNN(features)
    else:
        model = FeatureSNN(features)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    model.eval()
    outputs = []
    for f, raw, _, subject in DataLoader(dataset, 512, num_workers=4, pin_memory=True):
        logits, _ = model(f.to(device), raw.to(device), subject.to(device))
        outputs.append(logits.softmax(1).cpu().numpy())
    return np.concatenate(outputs), dataset.y, dataset.subject, dataset.start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", nargs="+",
                        help="kind,context,checkpoint; e.g. context,15,runs/x/best.pt")
    parser.add_argument("--output", type=Path, default=Path("runs/ensemble_metrics.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parsed = []
    for spec in args.spec:
        kind, context, checkpoint = spec.split(",", 2)
        parsed.append((kind, int(context), Path(checkpoint)))
    probabilities = {"val": [], "test": []}
    labels = {}
    metadata = {}
    for split in probabilities:
        for kind, context, checkpoint in parsed:
            p, y, subject, start = predict(root, split, kind, context, checkpoint, device)
            probabilities[split].append(p)
            labels[split] = y
            metadata[split] = (subject, start)

    if len(parsed) == 2:
        candidates = [(i / 20, 1 - i / 20) for i in range(21)]
    elif len(parsed) == 3:
        candidates = [(a / 10, b / 10, (10 - a - b) / 10)
                      for a in range(11) for b in range(11 - a)]
    elif len(parsed) == 4:
        resolution = 20
        candidates = [
            (a / resolution, b / resolution, c / resolution,
             (resolution - a - b - c) / resolution)
            for a in range(resolution + 1)
            for b in range(resolution + 1 - a)
            for c in range(resolution + 1 - a - b)
        ]
    else:
        # Resolution 0.1 gives 1,001 candidates for five experts and remains
        # cheap enough to audit exhaustively.
        candidates = list(simplex_grid(len(parsed), 10))
    rows = []
    for weights in candidates:
        combined = sum(w * p for w, p in zip(weights, probabilities["val"], strict=True))
        rows.append({"weights": weights, **score(labels["val"], combined)})
    best = max(rows, key=lambda r: (r["accuracy"], r["macro_f1"]))
    test_probability = sum(w * p for w, p in
                           zip(best["weights"], probabilities["test"], strict=True))
    result = {
        "models": [{"kind": k, "context": c, "checkpoint": str(p)} for k, c, p in parsed],
        "selected_on_validation": best,
        "test": score(labels["test"], test_probability),
        "candidates": rows,
    }
    val_probability = sum(w * p for w, p in
                          zip(best["weights"], probabilities["val"], strict=True))
    bias_rows = []
    val_log_probability = np.log(np.clip(val_probability, 1e-8, 1.0))
    for rest_bias in np.linspace(-0.8, 0.8, 65):
        calibrated = val_log_probability.copy()
        calibrated[:, 0] += rest_bias
        bias_rows.append({"rest_logit_bias": float(rest_bias),
                          **score(labels["val"], calibrated)})
    best_bias = max(bias_rows, key=lambda r: (r["accuracy"], r["macro_f1"]))
    test_calibrated = np.log(np.clip(test_probability, 1e-8, 1.0))
    test_calibrated[:, 0] += best_bias["rest_logit_bias"]
    result["rest_calibration"] = {
        "selected_on_validation": best_bias,
        "test": score(labels["test"], test_calibrated),
        "candidates": bias_rows,
    }
    sequence_rows = []
    for width in (1, 3, 5, 7, 9, 11, 15):
        candidate = smooth(
            val_probability, *metadata["val"], width=width, mode="causal"
        )
        sequence_rows.append({"width": width, **score(labels["val"], candidate)})
    sequence_best = max(sequence_rows, key=lambda r: (r["accuracy"], r["macro_f1"]))
    test_smoothed = smooth(
        test_probability, *metadata["test"], width=sequence_best["width"], mode="causal"
    )
    result["causal_sequence"] = {
        "selected_on_validation": sequence_best,
        "test": score(labels["test"], test_smoothed),
        "candidates": sequence_rows,
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
