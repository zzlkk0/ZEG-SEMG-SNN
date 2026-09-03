from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import DelayedResidualSNNLSTM, ResidualSNNLSTM, SNNLSTM


CLASS_NAMES = [
    "Rest",
    "Index flexion",
    "Index extension",
    "Middle flexion",
    "Middle extension",
    "Ring flexion",
    "Ring extension",
    "Little flexion",
    "Little extension",
    "Thumb adduction",
    "Thumb abduction",
    "Thumb flexion",
    "Thumb extension",
]


class NPZDataset(Dataset):
    def __init__(self, path: Path) -> None:
        data = np.load(path)
        self.x = data["x"]
        self.y = data["y"].astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.x[index].astype(np.float32)),
            torch.tensor(self.y[index], dtype=torch.long),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(labels: np.ndarray, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(labels, minlength=13).astype(np.float64)
    if mode == "sqrt":
        weights = np.sqrt(counts.sum() / counts)
    elif mode == "inverse":
        weights = counts.sum() / counts
    else:
        raise ValueError(mode)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int = 0,
) -> dict:
    model.eval()
    losses, predictions, targets = [], [], []
    spike_rates = []
    for index, (x, y) in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, diagnostics = model(x)
        losses.append(float(criterion(logits, y)))
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(y.cpu().tolist())
        spike_rates.append(
            [
                float(diagnostics["lif1_spike_rate"]),
                float(diagnostics["lif2_spike_rate"]),
            ]
        )
    rates = np.asarray(spike_rates).mean(axis=0)
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
        "spike_rates": rates.tolist(),
        "sparsity": float(1.0 - rates.mean()),
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=list(range(13))
        ).tolist(),
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int = 0,
) -> float:
    model.train()
    losses = []
    for index, (x, y) in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=root / "data" / "processed"
    )
    parser.add_argument("--run-dir", type=Path, default=root / "runs" / "membrane")
    parser.add_argument(
        "--frontend-checkpoint",
        type=Path,
        default=root.parent
        / "semg_snn_fpga_reproduction"
        / "runs"
        / "full_baseline"
        / "best.pt",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--feature-mode", choices=["membrane", "spike", "both"], default="membrane"
    )
    parser.add_argument(
        "--class-weights", choices=["none", "sqrt", "inverse"], default="none"
    )
    parser.add_argument(
        "--architecture",
        choices=["direct", "residual", "delayed_residual"],
        default="direct",
    )
    parser.add_argument("--freeze-snn", action="store_true")
    parser.add_argument("--snn-scale", type=float, default=20.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    train_set = NPZDataset(args.data_dir / "train.npz")
    val_set = NPZDataset(args.data_dir / "val.npz")
    test_set = NPZDataset(args.data_dir / "test.npz")
    print(
        f"dataset train={len(train_set)} val={len(val_set)} test={len(test_set)}",
        flush=True,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    if args.architecture == "delayed_residual":
        model = DelayedResidualSNNLSTM(
            decay=args.decay,
            threshold=args.threshold,
            lstm_hidden=args.lstm_hidden,
            dropout=args.dropout,
            snn_scale=args.snn_scale,
        )
    elif args.architecture == "residual":
        model = ResidualSNNLSTM(
            decay=args.decay,
            threshold=args.threshold,
            lstm_hidden=args.lstm_hidden,
            dropout=args.dropout,
            snn_scale=args.snn_scale,
        )
    else:
        model = SNNLSTM(
            decay=args.decay,
            threshold=args.threshold,
            lstm_hidden=args.lstm_hidden,
            dropout=args.dropout,
            feature_mode=args.feature_mode,
        )
    if args.frontend_checkpoint and args.frontend_checkpoint.exists():
        checkpoint = torch.load(
            args.frontend_checkpoint, map_location="cpu", weights_only=False
        )
        model.load_snn_frontend(checkpoint)
        print(f"loaded SNN frontend from {args.frontend_checkpoint}", flush=True)
    if args.freeze_snn:
        if not hasattr(model, "freeze_snn"):
            for layer in (model.lif1, model.lif2):
                for parameter in layer.parameters():
                    parameter.requires_grad = False
        else:
            model.freeze_snn()
    model.to(device)

    weights = class_weights(train_set.y, args.class_weights)
    criterion = nn.CrossEntropyLoss(
        weight=None if weights is None else weights.to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    print(
        f"parameters={sum(p.numel() for p in model.parameters())} "
        f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)} "
        f"architecture={args.architecture} feature_mode={args.feature_mode} "
        f"class_weights={args.class_weights}",
        flush=True,
    )

    best_accuracy = -1.0
    best_loss = float("inf")
    stale = 0
    history = []
    best_path = args.run_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.max_train_batches,
        )
        val = evaluate(
            model, val_loader, criterion, device, args.max_eval_batches
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "spike_rates": val["spike_rates"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        improved = (
            val["accuracy"] > best_accuracy
            or (
                val["accuracy"] == best_accuracy
                and val["loss"] < best_loss
            )
        )
        if improved:
            best_accuracy = val["accuracy"]
            best_loss = val["loss"]
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    test = evaluate(model, test_loader, criterion, device)
    result = {
        "best_epoch": int(state["epoch"]),
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "class_names": CLASS_NAMES,
        "test": test,
        "history": history,
        "args": {key: str(value) for key, value in vars(args).items()},
    }
    (args.run_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "best_epoch": state["epoch"],
                "test_accuracy": test["accuracy"],
                "test_macro_f1": test["macro_f1"],
                "test_spike_rates": test["spike_rates"],
                "test_sparsity": test["sparsity"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
