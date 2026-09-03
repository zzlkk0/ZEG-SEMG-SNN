"""QAT fine-tuning for the hardware-friendly Context/Hybrid experts.

Mirrors train.py's data pipeline and evaluation, but trains the HW model
graphs from hw_model.py (fake-quant weights/activations, no LayerNorm/GELU/
float division) starting from a warm-start of the existing FP32 checkpoints.

See ../../docs/specifications/RETRAIN_FOR_FPGA_SPEC.md for the deployment target.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader

from train import EMGDataset
from hw_model import (
    HWClassAdaptiveContextSNN,
    HWHybridSNN,
    remap_context_state,
    remap_hybrid_state,
)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions, targets, subjects, rates = [], [], [], []
    for f, raw, y, subject in loader:
        output, batch_rates = model(f.to(device), raw.to(device), subject.to(device))
        predictions.extend(output.argmax(1).cpu().tolist())
        targets.extend(y.tolist())
        subjects.extend(subject.tolist())
        rates.append([float(v) for v in batch_rates])
    result = {
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro"),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=range(13)).tolist(),
        "rates": np.mean(rates, axis=0).tolist(),
    }
    y, p, s = np.asarray(targets), np.asarray(predictions), np.asarray(subjects)
    result["gesture_accuracy"] = float(np.mean(p[y != 0] == y[y != 0]))
    result["subject_accuracy"] = {
        str(i + 1): float(np.mean(p[s == i] == y[s == i])) for i in range(10)
    }
    return result


def build_model(name: str, feature_count: int, weight_bits: int, act_frac_bits: int, act_int_bits: int):
    if name == "context_hw":
        return HWClassAdaptiveContextSNN(
            feature_count,
            weight_bits=weight_bits,
            act_frac_bits=act_frac_bits,
            act_int_bits=act_int_bits,
        )
    if name == "hybrid_hw":
        return HWHybridSNN(
            feature_count,
            weight_bits=weight_bits,
            act_frac_bits=act_frac_bits,
            act_int_bits=act_int_bits,
        )
    raise ValueError(f"unknown model {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--model", choices=["context_hw", "hybrid_hw"], required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-power", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument(
        "--selection-metric",
        choices=("accuracy", "macro_f1", "gesture_accuracy"),
        default="accuracy",
    )
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--act-frac-bits", type=int, default=8)
    parser.add_argument("--act-int-bits", type=int, default=8)
    parser.add_argument("--strong-raw-augment", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--init", type=Path, required=True, help="FP32 warm-start checkpoint")
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = args.data_dir or (root / "data")

    is_context = args.model == "context_hw"
    dataset_kwargs = dict(
        context=23 if is_context else 1,
        continuous_context=is_context,
        stream_context=is_context,
    )
    sets = {
        s: EMGDataset(
            data / f"{s}.npz",
            data / "normalization.npz",
            augment=(s == "train"),
            strong_raw_augment=args.strong_raw_augment and s == "train",
            **dataset_kwargs,
        )
        for s in ("train", "val", "test")
    }
    loaders = {
        "train": DataLoader(
            sets["train"], args.batch_size, shuffle=True, num_workers=args.workers,
            pin_memory=True, persistent_workers=args.workers > 0,
        ),
        "val": DataLoader(sets["val"], args.batch_size * 2, num_workers=args.workers, pin_memory=True),
        "test": DataLoader(sets["test"], args.batch_size * 2, num_workers=args.workers, pin_memory=True),
    }
    feature_count = sets["train"].features.shape[1]

    model = build_model(
        args.model, feature_count, args.weight_bits, args.act_frac_bits, args.act_int_bits
    ).to(device)

    source = torch.load(args.init, map_location=device, weights_only=False)["model"]
    remap = remap_context_state if is_context else remap_hybrid_state
    remapped = remap(source)
    compatible = {
        k: v for k, v in remapped.items()
        if k in model.state_dict() and model.state_dict()[k].shape == v.shape
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(
        f"warm-start={args.init} loaded={len(compatible)} missing={missing} unexpected={unexpected}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    counts = np.bincount(sets["train"].y, minlength=13)
    weights = torch.tensor(
        (counts.sum() / (13 * counts)) ** args.weight_power, dtype=torch.float32, device=device
    )

    run = root / "runs" / args.run_name
    run.mkdir(parents=True, exist_ok=True)
    best, stale, history = -1.0, 0, []
    print(f"device={device} model={args.model} features={feature_count}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for f, raw, y, subject in loaders["train"]:
            f, raw, y, subject = f.to(device), raw.to(device), y.to(device), subject.to(device)
            optimizer.zero_grad(set_to_none=True)
            output, _ = model(f, raw, subject)
            loss = nn.functional.cross_entropy(
                output, y, weight=weights, label_smoothing=args.label_smoothing
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        val = evaluate(model, loaders["val"], device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **val}
        history.append(row)
        print(
            json.dumps({k: v for k, v in row.items() if k not in ("confusion_matrix", "subject_accuracy")}),
            flush=True,
        )
        selection_value = val[args.selection_metric]
        if selection_value > best:
            best, stale = selection_value, 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "args": vars(args), "validation": val},
                run / "best.pt",
            )
        else:
            stale += 1
        if stale >= args.patience:
            break

    ckpt = torch.load(run / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    serializable_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    result = {
        "best_epoch": ckpt["epoch"],
        "args": serializable_args,
        "validation": evaluate(model, loaders["val"], device),
        "test": evaluate(model, loaders["test"], device),
        "history": history,
    }
    (run / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        "FINAL " + json.dumps({k: v for k, v in result.items() if k not in ("history", "args")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
