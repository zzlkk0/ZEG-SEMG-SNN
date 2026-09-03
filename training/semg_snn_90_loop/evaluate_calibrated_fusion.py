from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from evaluate_ensemble import predict, score


class ClasswiseConvexFusion(torch.nn.Module):
    """A tiny calibrated readout over frozen SNN probabilities."""

    def __init__(self, models: int, classes: int = 13):
        super().__init__()
        self.weight_logits = torch.nn.Parameter(torch.zeros(models, classes))
        self.bias = torch.nn.Parameter(torch.zeros(classes))

    def forward(self, probability):
        weights = self.weight_logits.softmax(dim=0)
        mixture = (weights[None] * probability).sum(dim=1).clamp_min(1e-7)
        return mixture.log() + self.bias


def fit(x, y, regularization, steps, device):
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    model = ClasswiseConvexFusion(x.shape[1], x.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        weights = model.weight_logits.softmax(dim=0)
        uniform = torch.full_like(weights, 1.0 / weights.shape[0])
        penalty = (weights - uniform).square().mean() + 0.1 * model.bias.square().mean()
        loss = torch.nn.functional.cross_entropy(logits, y) + regularization * penalty
        loss.backward()
        optimizer.step()
    return model


@torch.no_grad()
def infer(model, x, device):
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    return model(x).softmax(dim=1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", nargs="+")
    parser.add_argument(
        "--output", type=Path, default=Path("runs/calibrated_fusion_metrics.json")
    )
    parser.add_argument("--cv-steps", type=int, default=250)
    parser.add_argument("--final-steps", type=int, default=600)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parsed = []
    for spec in args.spec:
        kind, context, checkpoint = spec.split(",", 2)
        parsed.append((kind, int(context), Path(checkpoint)))

    probability, labels, subjects = {}, {}, {}
    for split in ("val", "test"):
        parts = []
        for kind, context, checkpoint in parsed:
            p, y, subject, _ = predict(
                root, split, kind, context, checkpoint, device
            )
            parts.append(p)
            if split in labels:
                if not np.array_equal(labels[split], y):
                    raise RuntimeError("model predictions are not label-aligned")
            labels[split], subjects[split] = y, subject
        probability[split] = np.stack(parts, axis=1)

    val_probability = np.clip(probability["val"], 1e-7, 1.0)
    test_probability = np.clip(probability["test"], 1e-7, 1.0)
    regularizations = (0.0, 0.01, 0.1, 1.0, 10.0)
    cv_rows = []
    # Five subject-disjoint folds avoid leakage from heavily overlapping windows.
    for regularization in regularizations:
        fold_predictions, fold_targets = [], []
        for fold in range(5):
            held_subjects = (2 * fold, 2 * fold + 1)
            held = np.isin(subjects["val"], held_subjects)
            model = fit(
                val_probability[~held], labels["val"][~held], regularization,
                args.cv_steps, device,
            )
            fold_predictions.append(infer(model, val_probability[held], device))
            fold_targets.append(labels["val"][held])
        p = np.concatenate(fold_predictions)
        y = np.concatenate(fold_targets)
        cv_rows.append({"regularization": regularization, **score(y, p)})
    selected = max(cv_rows, key=lambda row: (row["accuracy"], row["macro_f1"]))
    model = fit(
        val_probability, labels["val"], selected["regularization"],
        args.final_steps, device,
    )
    validation_probability = infer(model, val_probability, device)
    calibrated_test_probability = infer(model, test_probability, device)
    weights = model.weight_logits.softmax(dim=0).detach().cpu().numpy()
    result = {
        "models": [
            {"kind": kind, "context": context, "checkpoint": str(checkpoint)}
            for kind, context, checkpoint in parsed
        ],
        "protocol": {
            "hyperparameter_selection": "5-fold subject-disjoint CV within repetition 3",
            "calibration_fit": "full repetition 3",
            "test": "untouched repetition 5",
            "parameters": int(weights.size + model.bias.numel()),
        },
        "cross_validation": {"selected": selected, "candidates": cv_rows},
        "validation": score(labels["val"], validation_probability),
        "test": score(labels["test"], calibrated_test_probability),
        "classwise_model_weights": weights.tolist(),
        "class_bias": model.bias.detach().cpu().tolist(),
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
