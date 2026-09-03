from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from model import ContextSNN
from train import EMGDataset


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probabilities, targets = [], []
    for f, raw, y, subject in loader:
        logits, _ = model(f.to(device), raw.to(device), subject.to(device))
        probabilities.append(logits.softmax(1).cpu().numpy())
        targets.extend(y.tolist())
    return np.concatenate(probabilities), np.asarray(targets)


def score(probability, targets):
    prediction = probability.argmax(1)
    return {
        "accuracy": accuracy_score(targets, prediction),
        "macro_f1": f1_score(targets, prediction, average="macro"),
        "gesture_accuracy": float(np.mean(prediction[targets != 0] == targets[targets != 0])),
    }


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--context", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = root / "data"
    datasets = {
        split: EMGDataset(data / f"{split}.npz", data / "normalization.npz",
                          split == "train", context=args.context)
        for split in ("train", "val", "test")
    }
    initial = torch.load(args.init, map_location=device, weights_only=False)["model"]
    run = root / "runs" / f"per_subject_context{args.context}"
    run.mkdir(parents=True, exist_ok=True)
    subject_results = {}
    all_test_probability = []
    all_test_targets = []
    for subject in range(10):
        indices = {
            split: np.flatnonzero(dataset.subject == subject).tolist()
            for split, dataset in datasets.items()
        }
        loaders = {
            split: DataLoader(Subset(datasets[split], idx), args.batch_size,
                              shuffle=split == "train", num_workers=2, pin_memory=True)
            for split, idx in indices.items()
        }
        model = ContextSNN(datasets["train"].features.shape[1]).to(device)
        model.load_state_dict(initial)
        val_probability, val_targets = predict(model, loaders["val"], device)
        best = score(val_probability, val_targets)
        best_epoch = 0
        checkpoint = run / f"subject_{subject + 1}.pt"
        torch.save({"model": model.state_dict(), "epoch": 0, "validation": best}, checkpoint)
        counts = np.bincount(datasets["train"].y[indices["train"]], minlength=13)
        weights = torch.tensor((counts.sum() / (13 * np.maximum(counts, 1))) ** 0.25,
                               device=device, dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
        stale = 0
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for f, raw, y, sid in loaders["train"]:
                f, raw, y, sid = f.to(device), raw.to(device), y.to(device), sid.to(device)
                optimizer.zero_grad(set_to_none=True)
                output, _ = model(f, raw, sid)
                loss = nn.functional.cross_entropy(output, y, weight=weights,
                                                    label_smoothing=0.02)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                losses.append(loss.item())
            probability, targets = predict(model, loaders["val"], device)
            validation = score(probability, targets)
            history.append({"epoch": epoch, "loss": float(np.mean(losses)), **validation})
            if validation["accuracy"] > best["accuracy"]:
                best, best_epoch, stale = validation, epoch, 0
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "validation": validation}, checkpoint)
            else:
                stale += 1
            if stale >= 4:
                break
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        test_probability, test_targets = predict(model, loaders["test"], device)
        test = score(test_probability, test_targets)
        subject_results[str(subject + 1)] = {
            "best_epoch": best_epoch, "validation": best, "test": test, "history": history
        }
        all_test_probability.append(test_probability)
        all_test_targets.append(test_targets)
        print(json.dumps({"subject": subject + 1, "best_epoch": best_epoch,
                          "validation": best, "test": test}), flush=True)
    combined_probability = np.concatenate(all_test_probability)
    combined_targets = np.concatenate(all_test_targets)
    result = {
        "protocol": "known-subject calibration; each model selected on repetition 3",
        "combined_test": score(combined_probability, combined_targets),
        "subjects": subject_results,
    }
    (run / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL " + json.dumps(result["combined_test"]), flush=True)


if __name__ == "__main__":
    main()
