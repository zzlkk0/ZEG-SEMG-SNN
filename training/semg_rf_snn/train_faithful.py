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

from faithful_model import FaithfulRFSNN


class Dataset52(Dataset):
    def __init__(self, path: Path, augment: bool):
        data = np.load(path)
        self.x = data["x"]
        self.y = data["y"].astype(np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.x[index].astype(np.float32)
        if self.augment:
            # Paper augmentation: delay/anticipate each recording by -8..8 steps.
            shift = np.random.randint(-8, 9)
            if shift > 0:
                x = np.concatenate((np.zeros_like(x[:shift]), x[:-shift]))
            elif shift < 0:
                x = np.concatenate((x[-shift:], np.zeros_like(x[: -shift])))
        return torch.from_numpy(x), int(self.y[index])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    pred, truth, rates = [], [], []
    for x, y in loader:
        output, layer_rates = model(x.to(device, non_blocking=True))
        pred.extend(output.argmax(1).cpu().tolist())
        truth.extend(y.tolist())
        rates.append([float(v) for v in layer_rates])
    return {
        "accuracy": accuracy_score(truth, pred),
        "macro_f1": f1_score(truth, pred, average="macro"),
        "rates": np.mean(rates, axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--protocol", choices=["paper", "strict"], required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rf-neurons", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = root / "data" / f"faithful_{args.protocol}"
    train_set = Dataset52(data / "train.npz", True)
    val_set = Dataset52(data / "val.npz", False)
    loaders = {
        "train": DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers,
                            pin_memory=True, persistent_workers=args.workers > 0),
        "val": DataLoader(val_set, args.batch_size * 2, num_workers=args.workers, pin_memory=True),
    }
    test_path = data / "test.npz"
    if test_path.exists():
        loaders["test"] = DataLoader(Dataset52(test_path, False), args.batch_size * 2,
                                     num_workers=args.workers, pin_memory=True)
    model = FaithfulRFSNN(args.rf_neurons).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    counts = np.bincount(train_set.y, minlength=13)
    # Paper's inverse-frequency balancing with a softened gamma to retain rest accuracy.
    weights = torch.tensor((counts.sum() / (13 * counts)) ** 0.5, device=device, dtype=torch.float32)
    run = root / "runs" / f"faithful_{args.protocol}_rf{args.rf_neurons}"
    run.mkdir(parents=True, exist_ok=True)
    best, stale, history = -1.0, 0, []
    print(f"device={device} protocol={args.protocol} train={len(train_set)} val={len(val_set)}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output, _ = model(x)
            loss = nn.functional.cross_entropy(output, y, weight=weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        val = evaluate(model, loaders["val"], device)
        row = {"epoch": epoch, "train_loss": np.mean(losses), **val}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val["accuracy"] > best:
            best, stale = val["accuracy"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val}, run / "best.pt")
        else:
            stale += 1
        if stale >= args.patience:
            break
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result = {"protocol": args.protocol, "best_epoch": checkpoint["epoch"],
              "validation": evaluate(model, loaders["val"], device), "history": history}
    if "test" in loaders:
        result["test"] = evaluate(model, loaders["test"], device)
    (run / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL " + json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True)


if __name__ == "__main__":
    main()
