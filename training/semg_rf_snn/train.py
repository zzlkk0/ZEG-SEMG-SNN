from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import RFSNN


class RawDataset(Dataset):
    def __init__(self, path: Path, norm: Path, augment: bool = False):
        data = np.load(path)
        stats = np.load(norm)
        self.x = data["x"].astype(np.float32)
        self.y = data["y"].astype(np.int64)
        self.median = stats["median"].astype(np.float32)
        self.scale = stats["scale"].astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = (self.x[i] - self.median) / self.scale
        if self.augment:
            x = x * np.random.uniform(0.9, 1.1, (1, 16)).astype(np.float32)
            x = x + np.random.normal(0, 0.015, x.shape).astype(np.float32)
        return torch.from_numpy(x), int(self.y[i])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    predictions, targets, losses, rates = [], [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output, layer_rates = model(x)
        losses.append(nn.functional.cross_entropy(output, y).item())
        predictions.extend(output.argmax(1).cpu().tolist())
        targets.extend(y.cpu().tolist())
        rates.append([float(v) for v in layer_rates])
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro"),
        "rates": np.mean(rates, axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=root / "runs" / "strict_split")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = root / "data"
    train = RawDataset(data / "train_raw.npz", data / "normalization.npz", True)
    val = RawDataset(data / "val_raw.npz", data / "normalization.npz")
    test = RawDataset(data / "test_raw.npz", data / "normalization.npz")
    loaders = {
        "train": DataLoader(train, args.batch_size, shuffle=True, num_workers=args.workers,
                            pin_memory=True, persistent_workers=args.workers > 0),
        "val": DataLoader(val, args.batch_size * 2, num_workers=args.workers, pin_memory=True),
        "test": DataLoader(test, args.batch_size * 2, num_workers=args.workers, pin_memory=True),
    }
    model = RFSNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    counts = np.bincount(train.y, minlength=13)
    weights = torch.tensor((counts.sum() / (13 * counts)) ** 0.35, device=device, dtype=torch.float32)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    best, stale, history = -1.0, 0, []
    print(f"device={device} train={len(train)} val={len(val)} test={len(test)}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                output, _ = model(x)
                loss = nn.functional.cross_entropy(output, y, weight=weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            batch_losses.append(loss.item())
        scheduler.step()
        metrics = evaluate(model, loaders["val"], device)
        row = {"epoch": epoch, "train_loss": np.mean(batch_losses), **metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["accuracy"] > best:
            best, stale = metrics["accuracy"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": metrics}, args.run_dir / "best.pt")
        else:
            stale += 1
        if stale >= args.patience:
            break
    state = torch.load(args.run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    result = {"best_epoch": state["epoch"], "validation": evaluate(model, loaders["val"], device),
              "test": evaluate(model, loaders["test"], device), "history": history}
    (args.run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL " + json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True)


if __name__ == "__main__":
    main()
