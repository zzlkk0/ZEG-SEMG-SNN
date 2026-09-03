from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_ensemble import predict, score, simplex_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", nargs="+")
    parser.add_argument(
        "--output", type=Path, default=Path("runs/subject_calibrated_ensemble_metrics.json")
    )
    parser.add_argument("--resolution", type=int, default=20)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parsed = []
    for spec in args.spec:
        kind, context, checkpoint = spec.split(",", 2)
        parsed.append((kind, int(context), Path(checkpoint)))

    probability, labels, subjects = {}, {}, {}
    for split in ("val", "test"):
        pieces = []
        for kind, context, checkpoint in parsed:
            p, y, subject, _ = predict(
                root, split, kind, context, checkpoint, device
            )
            pieces.append(p)
            if split in labels and not np.array_equal(labels[split], y):
                raise RuntimeError("predictions are not aligned")
            labels[split], subjects[split] = y, subject
        probability[split] = pieces

    grid = list(simplex_grid(len(parsed), args.resolution))
    raw_test = np.zeros_like(probability["test"][0])
    calibrated_test = np.zeros_like(raw_test)
    rows = []
    for subject in range(10):
        val_mask = subjects["val"] == subject
        test_mask = subjects["test"] == subject
        candidates = []
        for weights in grid:
            p = sum(
                weight * model_probability[val_mask]
                for weight, model_probability in zip(
                    weights, probability["val"], strict=True
                )
            )
            candidates.append({"weights": weights, **score(labels["val"][val_mask], p)})
        best = max(candidates, key=lambda row: (row["accuracy"], row["macro_f1"]))
        val_probability = sum(
            weight * model_probability[val_mask]
            for weight, model_probability in zip(
                best["weights"], probability["val"], strict=True
            )
        )
        test_probability = sum(
            weight * model_probability[test_mask]
            for weight, model_probability in zip(
                best["weights"], probability["test"], strict=True
            )
        )
        raw_test[test_mask] = test_probability
        bias_candidates = []
        val_log = np.log(np.clip(val_probability, 1e-8, 1.0))
        for rest_bias in np.linspace(-0.8, 0.8, 65):
            adjusted = val_log.copy()
            adjusted[:, 0] += rest_bias
            bias_candidates.append(
                {"rest_logit_bias": float(rest_bias),
                 **score(labels["val"][val_mask], adjusted)}
            )
        best_bias = max(
            bias_candidates, key=lambda row: (row["accuracy"], row["macro_f1"])
        )
        adjusted_test = np.log(np.clip(test_probability, 1e-8, 1.0))
        adjusted_test[:, 0] += best_bias["rest_logit_bias"]
        calibrated_test[test_mask] = torch.from_numpy(adjusted_test).softmax(1).numpy()
        rows.append({
            "subject": subject + 1,
            "validation_selection": best,
            "rest_calibration": best_bias,
            "test_raw": score(labels["test"][test_mask], test_probability),
            "test_calibrated": score(
                labels["test"][test_mask], calibrated_test[test_mask]
            ),
        })

    result = {
        "models": [
            {"kind": kind, "context": context, "checkpoint": str(checkpoint)}
            for kind, context, checkpoint in parsed
        ],
        "protocol": (
            "Per-subject weights and rest bias selected only on repetition 3; "
            "applied unchanged to repetition 5."
        ),
        "test_raw": score(labels["test"], raw_test),
        "test_calibrated": score(labels["test"], calibrated_test),
        "subjects": rows,
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
