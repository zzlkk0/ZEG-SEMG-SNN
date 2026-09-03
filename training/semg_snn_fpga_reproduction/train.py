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

from model import PaperSNN, PaperSNNWithDelays, spike_rate_loss


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
        self.subject = data["subject"]
        self.start = data["start"]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.x[index].astype(np.float32)),
            torch.tensor(self.y[index], dtype=torch.long),
            torch.tensor(self.subject[index], dtype=torch.long),
            torch.tensor(self.start[index], dtype=torch.long),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> dict:
    model.eval()
    losses = []
    predictions = []
    targets = []
    subjects = []
    starts = []
    layer_rates = []

    for batch_index, (x, y, subject, start) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        output, rates = model(x)
        loss = spike_rate_loss(output, y)
        prediction = output.sum(dim=1).argmax(dim=1)

        losses.append(float(loss))
        predictions.extend(prediction.cpu().tolist())
        targets.extend(y.cpu().tolist())
        subjects.extend(subject.tolist())
        starts.extend(start.tolist())
        layer_rates.append([float(rate) for rate in rates])

    layer_rates_array = np.asarray(layer_rates)
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=list(range(13))
        ).tolist(),
        "layer_spike_rates": layer_rates_array.mean(axis=0).tolist(),
        "sparsity": float(1.0 - layer_rates_array.mean()),
        "predictions": predictions,
        "targets": targets,
        "subjects": subjects,
        "starts": starts,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int = 0,
) -> float:
    model.train()
    losses = []
    for batch_index, (x, y, _, _) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output, _ = model(x)
        loss = spike_rate_loss(output, y)
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
    parser.add_argument("--run-dir", type=Path, default=root / "runs" / "baseline")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--max-delay", type=int, default=0)
    parser.add_argument("--initial-delay", type=float, default=1.0)
    parser.add_argument("--init-checkpoint", type=Path)
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

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=generator,
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

    if args.max_delay > 0:
        model = PaperSNNWithDelays(
            args.decay,
            args.threshold,
            max_delay=args.max_delay,
            initial_delay=args.initial_delay,
        ).to(device)
    else:
        model = PaperSNN(args.decay, args.threshold).to(device)
    if args.init_checkpoint:
        initial_state = torch.load(
            args.init_checkpoint, map_location=device, weights_only=False
        )
        missing, unexpected = model.load_state_dict(
            initial_state["model"], strict=False
        )
        print(
            f"initialized from {args.init_checkpoint}; "
            f"missing={missing}; unexpected={unexpected}",
            flush=True,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters={parameters}", flush=True)

    best_loss = float("inf")
    stale_epochs = 0
    history = []
    checkpoint = args.run_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.max_train_batches,
        )
        validation = evaluate(
            model, val_loader, device, args.max_eval_batches
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
            "val_macro_f1": validation["macro_f1"],
            "val_layer_spike_rates": validation["layer_spike_rates"],
        }
        if hasattr(model, "delay_statistics"):
            row["delay_statistics"] = model.delay_statistics()
        history.append(row)
        print(json.dumps(row), flush=True)

        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                },
                checkpoint,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    test = evaluate(model, test_loader, device, args.max_eval_batches)
    result = {
        "best_epoch": state["epoch"],
        "device": str(device),
        "parameters": parameters,
        "class_names": CLASS_NAMES,
        "test": test,
        "history": history,
        "reproduction_gaps": [
            (
                "Axonal delays use linear interpolation in pure PyTorch rather "
                "than SLAYER's internal delay operator."
                if args.max_delay > 0
                else "No SLAYER trainable axonal delays in this baseline."
            ),
            "Paper does not specify LIF decay and threshold.",
            "Derivative implementation is not specified by the paper.",
            "Paper temporal voting is not applied to the primary raw-window metric.",
        ],
    }
    (args.run_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "test_accuracy": test["accuracy"],
                "test_macro_f1": test["macro_f1"],
                "test_sparsity": test["sparsity"],
                "test_layer_spike_rates": test["layer_spike_rates"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
