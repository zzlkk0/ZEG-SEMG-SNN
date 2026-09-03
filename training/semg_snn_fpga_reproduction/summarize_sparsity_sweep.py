"""Aggregate the sparsity Pareto sweep's per-lambda runs into one table.

Reads runs/sparsity_lambda_*/metrics.json (written by train_sparsity.py) and
prints/writes a single comparison table: lambda_sparsity -> test accuracy,
macro-F1, gesture-accuracy, sparsity(group=1), vs the un-regularized baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINE_ACCURACY = 0.8393047179851011
BASELINE_MACRO_F1 = 0.6561095888360277
BASELINE_GESTURE_ACCURACY = 0.5794324520979869
# freshly re-measured via evaluate(delay62_finetune, test set) -- see
# DELAY_RETRAIN_REPORT.md for the exact command; do not hand-edit these.
BASELINE_LAYER_SPIKE_RATES = (0.12808612800306743, 0.06820049939884079, 0.06654793582856655, 0.039364894727865855)
BASELINE_SPARSITY = 1.0 - (
    sum(rate * size for rate, size in zip(BASELINE_LAYER_SPIKE_RATES, (64, 128, 64, 13))) / (64 + 128 + 64 + 13)
)


def main() -> None:
    rows = []
    for run_dir in sorted(ROOT.glob("runs/sparsity_lambda_*")):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        data = json.loads(metrics_path.read_text())
        test = data["test"]
        rows.append({
            "lambda_sparsity": data["lambda_sparsity"],
            "run_dir": str(run_dir),
            "best_epoch": data["best_epoch"],
            "accuracy": test["accuracy"],
            "macro_f1": test["macro_f1"],
            "gesture_accuracy": test["gesture_accuracy"],
            "sparsity_group1": test["sparsity_group1"],
        })
    rows.sort(key=lambda r: r["lambda_sparsity"])

    print(f"{'lambda':>10s}{'accuracy':>12s}{'macro_f1':>12s}{'gesture_acc':>14s}{'sparsity(g=1)':>16s}{'best_epoch':>12s}")
    print(f"{'baseline':>10s}{BASELINE_ACCURACY:12.4f}{BASELINE_MACRO_F1:12.4f}"
          f"{BASELINE_GESTURE_ACCURACY:14.4f}{BASELINE_SPARSITY:16.4f}{'--':>12s}")
    for row in rows:
        print(f"{row['lambda_sparsity']:10.4f}{row['accuracy']:12.4f}{row['macro_f1']:12.4f}"
              f"{row['gesture_accuracy']:14.4f}{row['sparsity_group1']:16.4f}{row['best_epoch']:12d}")

    output = {
        "baseline": {
            "accuracy": BASELINE_ACCURACY, "macro_f1": BASELINE_MACRO_F1,
            "gesture_accuracy": BASELINE_GESTURE_ACCURACY, "sparsity_group1": BASELINE_SPARSITY,
        },
        "sweep": rows,
    }
    out_path = ROOT / "runs" / "sparsity_sweep_summary.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
