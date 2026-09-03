"""Evaluate the Delay-SNN baseline with and without temporal voting.

Reports, on the full NinaPro DB5 E1 repetition-5 strict test split
(11,276 windows):
  - strict per-window accuracy (no post-processing) -- the primary number
    used everywhere else in this project (83.9305%).
  - paper_style_vote accuracy (1-window-lookahead debounce, see voting.py).
  - two_window_debounce_vote accuracy (causal debounce, already used by
    evaluate_checkpoint.py, included for comparison).
  - optional +-200ms tolerance accuracy (best-effort reconstruction; the
    paper does not publish exact pseudocode for this metric, so the
    assumption used here is stated explicitly in the output).

No training happens in this script; it only runs inference + post-process.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from model import PaperSNNWithDelays
from train import NPZDataset, evaluate
from voting import paper_style_vote, two_window_debounce_vote


def compact(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro")),
        "gesture_accuracy": float(np.mean(p[y != 0] == y[y != 0])),
        "correct": int(np.sum(p == y)),
        "total": int(len(y)),
    }


def tolerance_200ms(
    predictions: np.ndarray,
    targets: np.ndarray,
    subjects: np.ndarray,
    starts: np.ndarray,
    hop_ms: int = 100,
    tolerance_ms: int = 200,
) -> dict:
    """Best-effort +-200ms tolerance metric.

    Assumption (paper does not publish exact pseudocode, per spec 2.1):
    a window's prediction counts as correct if it matches the true label of
    *any* window within +-tolerance_ms of it, for the same subject, ordered
    by start sample. This is deliberately a generous, best-effort
    reconstruction and is reported only as a supplementary number.
    """
    radius = tolerance_ms // hop_ms
    correct = np.zeros(len(predictions), dtype=bool)
    for subject in np.unique(subjects):
        indexes = np.flatnonzero(subjects == subject)
        indexes = indexes[np.argsort(starts[indexes])]
        seq_targets = targets[indexes]
        seq_predictions = predictions[indexes]
        n = len(indexes)
        for local_i in range(n):
            lo = max(0, local_i - radius)
            hi = min(n, local_i + radius + 1)
            window_targets = seq_targets[lo:hi]
            correct[indexes[local_i]] = seq_predictions[local_i] in window_targets
    accuracy = float(correct.mean())
    return {
        "accuracy": accuracy,
        "correct": int(correct.sum()),
        "total": int(len(predictions)),
        "assumption": (
            f"prediction counts correct if it matches the true label of any "
            f"window within +-{tolerance_ms}ms ({radius} windows at {hop_ms}ms hop) "
            "of it, same subject -- paper does not publish exact pseudocode"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--checkpoint", type=Path, default=root / "runs" / "delay62_finetune" / "best.pt")
    parser.add_argument("--data", type=Path, default=root / "data" / "processed" / "test.npz")
    parser.add_argument("--output", type=Path, default=root / "runs" / "delay62_finetune" / "voting_metrics.json")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = state["args"]
    model = PaperSNNWithDelays(
        decay=float(saved_args["decay"]),
        threshold=float(saved_args["threshold"]),
        max_delay=int(saved_args["max_delay"]),
        initial_delay=float(saved_args.get("initial_delay", 1.0)),
    ).to(device)
    model.load_state_dict(state["model"])

    dataset = NPZDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    raw = evaluate(model, loader, device)

    predictions = np.asarray(raw["predictions"])
    targets = np.asarray(raw["targets"])
    subjects = np.asarray(raw["subjects"])
    starts = np.asarray(raw["starts"])

    strict = compact(targets, predictions)
    paper_voted = paper_style_vote(predictions, subjects, starts)
    paper_voted_metrics = compact(targets, paper_voted)
    causal_voted = two_window_debounce_vote(predictions, subjects, starts)
    causal_voted_metrics = compact(targets, causal_voted)
    tolerance_metrics = tolerance_200ms(predictions, targets, subjects, starts)

    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(state["epoch"]),
        "test_windows": int(len(targets)),
        "strict_per_window": strict,
        "paper_style_vote_1window_lookahead": paper_voted_metrics,
        "two_window_debounce_vote_causal": causal_voted_metrics,
        "tolerance_200ms_supplementary": tolerance_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
