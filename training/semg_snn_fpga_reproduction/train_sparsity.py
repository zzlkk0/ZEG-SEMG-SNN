"""Fine-tune PaperSNNWithDelays with an added spike-sparsity regularizer.

Warm-starts from an existing checkpoint (default: delay62_finetune, the
current deployed baseline) and adds `lambda_sparsity * mean(per-layer spike
rate)` to the training loss, on top of the existing spike_rate_loss. Topology,
weight/delay/decay/threshold formats are untouched -- only the training
objective changes, so the resulting checkpoint stays export-compatible with
the existing fixed-point deployment pipeline (export_fixed_point.py).

Used to produce a Pareto sweep of (lambda_sparsity -> accuracy, sparsity)
points, per DELAY_RETRAIN_TOWARD_PAPER_SPEC.md 2.4.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import PaperSNNWithDelays, spike_rate_loss
from train import NPZDataset, evaluate, set_seed


def sparsity_from_rates(mean_layer_rates: list[float], layer_sizes: tuple[int, ...]) -> float:
    """Group size = 1 (per-spike) sparsity, weighted by each layer's neuron count.

    sparsity = 1 - (total active spike-instances / total possible spike-instances)
    across all layers combined -- matches spec 2.4's Eq.4-style reporting with
    group size fixed at 1 (this deployment processes one synapse per cycle,
    not the paper's 4-wide grouped datapath).
    """
    total_size = sum(layer_sizes)
    weighted_rate = sum(rate * size for rate, size in zip(mean_layer_rates, layer_sizes)) / total_size
    return 1.0 - weighted_rate


LAYER_SIZES = (64, 128, 64, 13)


def train_epoch(model, loader, optimizer, device, lambda_sparsity):
    model.train()
    losses, class_losses, sparsity_losses = [], [], []
    for x, y, _, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output, rates = model(x)
        class_loss = spike_rate_loss(output, y)
        sparsity_penalty = sum(rates) / len(rates)  # mean spike rate across the 4 layers
        loss = class_loss + lambda_sparsity * sparsity_penalty
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        class_losses.append(float(class_loss.detach()))
        sparsity_losses.append(float(sparsity_penalty.detach()))
    return float(np.mean(losses)), float(np.mean(class_losses)), float(np.mean(sparsity_losses))


@torch.no_grad()
def evaluate_with_sparsity(model, loader, device, lambda_sparsity):
    base = evaluate(model, loader, device)
    sparsity = sparsity_from_rates(base["layer_spike_rates"], LAYER_SIZES)
    total_loss = base["loss"] + lambda_sparsity * (1.0 - sparsity)
    base["sparsity_group1"] = sparsity
    base["total_loss_with_sparsity_term"] = total_loss
    predictions = np.asarray(base["predictions"])
    targets = np.asarray(base["targets"])
    base["gesture_accuracy"] = float(
        np.mean(predictions[targets != 0] == targets[targets != 0])
    )
    return base


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "processed")
    parser.add_argument("--init-checkpoint", type=Path, default=root / "runs" / "delay62_finetune" / "best.pt")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--lambda-sparsity", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = NPZDataset(args.data_dir / "train.npz")
    val_set = NPZDataset(args.data_dir / "val.npz")
    test_set = NPZDataset(args.data_dir / "test.npz")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    init_state = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
    init_args = init_state["args"]
    model = PaperSNNWithDelays(
        decay=float(init_args["decay"]), threshold=float(init_args["threshold"]),
        max_delay=int(init_args["max_delay"]), initial_delay=float(init_args.get("initial_delay", 1.0)),
    ).to(device)
    model.load_state_dict(init_state["model"])
    print(f"warm-started from {args.init_checkpoint} (epoch {init_state['epoch']})", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    checkpoint_path = args.run_dir / "best.pt"
    best_loss, stale, history = float("inf"), 0, []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_class_loss, train_sparsity_loss = train_epoch(
            model, train_loader, optimizer, device, args.lambda_sparsity
        )
        validation = evaluate_with_sparsity(model, val_loader, device, args.lambda_sparsity)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_class_loss": train_class_loss,
            "train_mean_spike_rate": train_sparsity_loss,
            "val_loss_with_sparsity": validation["total_loss_with_sparsity_term"],
            "val_accuracy": validation["accuracy"],
            "val_sparsity_group1": validation["sparsity_group1"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        if validation["total_loss_with_sparsity_term"] < best_loss:
            best_loss, stale = validation["total_loss_with_sparsity_term"], 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch,
                 "args": {**vars(args), "decay": float(init_args["decay"]),
                          "threshold": float(init_args["threshold"]),
                          "data_dir": str(args.data_dir), "run_dir": str(args.run_dir),
                          "init_checkpoint": str(args.init_checkpoint)}},
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    test = evaluate_with_sparsity(model, test_loader, device, args.lambda_sparsity)
    result = {
        "lambda_sparsity": args.lambda_sparsity,
        "init_checkpoint": str(args.init_checkpoint),
        "best_epoch": state["epoch"],
        "test": test,
        "history": history,
    }
    (args.run_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print("FINAL " + json.dumps({
        "lambda_sparsity": args.lambda_sparsity,
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
        "test_gesture_accuracy": test["gesture_accuracy"],
        "test_sparsity_group1": test["sparsity_group1"],
    }), flush=True)


if __name__ == "__main__":
    main()
