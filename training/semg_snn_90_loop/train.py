from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from model import (
    ContextSNN,
    AdaptiveContextSNN,
    ClassAdaptiveContextSNN,
    ContextHybridSNN,
    FeatureSNN,
    HybridSNN,
    HybridMaxSNN,
    HybridSpatialSNN,
    SubjectFeatureSNN,
    SubjectFiLMContextSNN,
)


class EMGDataset(Dataset):
    def __init__(self, path: Path, normalization: Path, augment=False, context=1,
                 strong_raw_augment=False, reset_gaps=True,
                 continuous_context=False, stream_context=False):
        d, n = np.load(path), np.load(normalization)
        self.raw = d["raw"]
        self.features = d["features"]
        self.y = d["y"].astype(np.int64)
        self.subject = d["subject"].astype(np.int64)
        self.start = d["start"].astype(np.int64)
        self.fm, self.fs = n["feature_mean"], n["feature_std"]
        self.rm, self.rs = n["raw_mean"], n["raw_std"]
        self.augment = augment
        self.context = context
        self.strong_raw_augment = strong_raw_augment
        self.reset_gaps = reset_gaps
        self.continuous_context = continuous_context
        self.stream_context = stream_context
        if continuous_context:
            data_root = path.parent
            self.continuous_features = np.load(
                data_root / "continuous_features.npy", mmap_mode="r"
            )
            continuous_index = np.load(data_root / "continuous_index.npz")
            self.continuous_rep = continuous_index["repetition"]
            self.continuous_subject = continuous_index["subject"]
            self.continuous_offsets = continuous_index["offsets"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        subject = self.subject[i]
        if self.context > 1:
            if self.continuous_context:
                cursor = int(
                    self.continuous_offsets[subject] + (self.start[i] - 400) // 20
                )
                target_rep = int(self.continuous_rep[cursor])
                continuous_indices = []
                for _ in range(self.context):
                    continuous_indices.append(cursor)
                    previous = cursor - 1
                    same_subject = (
                        previous >= self.continuous_offsets[subject]
                        and self.continuous_subject[previous] == subject
                    )
                    previous_rep = (
                        int(self.continuous_rep[previous]) if same_subject else -1
                    )
                    if same_subject and (
                        self.stream_context or previous_rep in (0, target_rep)
                    ):
                        cursor = previous
                continuous_indices = continuous_indices[::-1]
                feature = (
                    self.continuous_features[continuous_indices].astype(np.float32)
                    - self.fm[subject]
                ) / self.fs[subject]
            else:
                indices = []
                cursor = i
                for _ in range(self.context):
                    indices.append(cursor)
                    contiguous = (
                        not self.reset_gaps
                        or self.start[cursor] - self.start[cursor - 1] == 20
                    )
                    if cursor > 0 and self.subject[cursor - 1] == subject and contiguous:
                        cursor -= 1
                indices = ([indices[-1]] * (self.context - len(indices)) + indices)[-self.context:]
                indices = indices[::-1]
                feature = (self.features[indices] - self.fm[subject]) / self.fs[subject]
        else:
            feature = (self.features[i] - self.fm[subject]) / self.fs[subject]
        feature = np.clip(feature, -8, 8).astype(np.float32)
        raw = (self.raw[i].astype(np.float32) - self.rm[subject]) / self.rs[subject]
        if self.augment:
            gain = np.random.uniform(0.9, 1.1, (1, 16)).astype(np.float32)
            raw = raw * gain + np.random.normal(0, 0.015, raw.shape).astype(np.float32)
            feature = feature + np.random.normal(0, 0.01, feature.shape).astype(np.float32)
            if self.strong_raw_augment:
                shift = np.random.randint(-8, 9)
                if shift > 0:
                    raw = np.concatenate((np.zeros_like(raw[:shift]), raw[:-shift]))
                elif shift < 0:
                    raw = np.concatenate((raw[-shift:], np.zeros_like(raw[:-shift])))
                signal_rms = float(np.sqrt(np.mean(raw * raw) + 1e-6))
                raw += np.random.normal(
                    0, signal_rms / np.sqrt(1000.0), raw.shape
                ).astype(np.float32)
                if np.random.random() < 0.2:
                    raw[:, np.random.randint(0, 16)] = 0
        return torch.from_numpy(feature), torch.from_numpy(raw), int(self.y[i]), int(subject)


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


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "--model",
        choices=[
            "feature", "subject", "context", "context_adaptive",
            "context_class_adaptive", "context_hybrid", "context_film",
            "hybrid", "hybrid_max", "hybrid_spatial"
        ],
        default="feature",
    )
    parser.add_argument("--context", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-power", type=float, default=0.25)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument(
        "--selection-metric",
        choices=("accuracy", "macro_f1", "gesture_accuracy"),
        default="accuracy",
    )
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--strong-raw-augment", action="store_true")
    parser.add_argument("--no-gap-reset", action="store_true")
    parser.add_argument("--continuous-context", action="store_true")
    parser.add_argument(
        "--stream-context",
        action="store_true",
        help="Use strictly causal continuous history without repetition boundaries.",
    )
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--init-secondary", type=Path)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = args.data_dir or (root / "data")
    sets = {s: EMGDataset(
                data / f"{s}.npz", data / "normalization.npz", s == "train",
                context=args.context if args.model in
                ("context", "context_adaptive", "context_class_adaptive",
                 "context_hybrid", "context_film") else 1,
                strong_raw_augment=args.strong_raw_augment and s == "train",
                reset_gaps=not args.no_gap_reset,
                continuous_context=args.continuous_context,
                stream_context=args.stream_context,
            )
            for s in ("train", "val", "test")}
    loaders = {
        "train": DataLoader(sets["train"], args.batch_size, shuffle=True, num_workers=args.workers,
                            pin_memory=True, persistent_workers=args.workers > 0),
        "val": DataLoader(sets["val"], args.batch_size * 2, num_workers=args.workers, pin_memory=True),
        "test": DataLoader(sets["test"], args.batch_size * 2, num_workers=args.workers, pin_memory=True),
    }
    feature_count = sets["train"].features.shape[1]
    if args.model == "feature":
        model = FeatureSNN(feature_count)
    elif args.model == "subject":
        model = SubjectFeatureSNN(feature_count)
    elif args.model == "context":
        model = ContextSNN(feature_count)
    elif args.model == "context_adaptive":
        model = AdaptiveContextSNN(feature_count)
    elif args.model == "context_class_adaptive":
        model = ClassAdaptiveContextSNN(feature_count)
    elif args.model == "context_hybrid":
        model = ContextHybridSNN(feature_count)
    elif args.model == "context_film":
        model = SubjectFiLMContextSNN(feature_count)
    elif args.model == "hybrid_max":
        model = HybridMaxSNN(feature_count)
    elif args.model == "hybrid_spatial":
        model = HybridSpatialSNN(feature_count)
    else:
        model = HybridSNN(feature_count)
    model = model.to(device)
    if args.init:
        source = torch.load(args.init, map_location=device, weights_only=False)["model"]
        if (
            args.model == "context_class_adaptive"
            and source.get("context_gamma_logit", torch.empty(0)).ndim == 0
        ):
            source["context_gamma_logit"] = source["context_gamma_logit"].repeat(13)
        compatible = {k: v for k, v in source.items()
                      if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        if args.model in ("hybrid_max", "hybrid_spatial") and "fuse.0.weight" in source:
            with torch.no_grad():
                model.fuse[0].weight.zero_()
                old = source["fuse.0.weight"]
                model.fuse[0].weight[:, : old.shape[1]].copy_(old)
        if args.model == "subject" and "out.weight" in source:
            with torch.no_grad():
                model.out_weight.copy_(source["out.weight"].unsqueeze(0).expand(10, -1, -1))
                model.out_bias.copy_(source["out.bias"].unsqueeze(0).expand(10, -1))
        print(f"initialized={args.init} loaded={len(compatible)} missing={missing} unexpected={unexpected}",
              flush=True)
    if args.init_secondary:
        secondary = torch.load(
            args.init_secondary, map_location=device, weights_only=False
        )["model"]
        compatible = {
            k: v for k, v in secondary.items()
            if k.startswith("conv.")
            and k in model.state_dict()
            and model.state_dict()[k].shape == v.shape
        }
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        print(
            f"secondary={args.init_secondary} loaded={len(compatible)} "
            f"missing={missing} unexpected={unexpected}",
            flush=True,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    counts = np.bincount(sets["train"].y, minlength=13)
    weights = torch.tensor((counts.sum() / (13 * counts)) ** args.weight_power,
                           dtype=torch.float32, device=device)
    run = root / "runs" / args.run_name
    run.mkdir(parents=True, exist_ok=True)
    best, stale, history = -1.0, 0, []
    print(f"device={device} model={args.model} features={feature_count}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for f, raw, y, subject in loaders["train"]:
            f, raw, y, subject = f.to(device), raw.to(device), y.to(device), subject.to(device)
            optimizer.zero_grad(set_to_none=True)
            if args.mixup_alpha > 0:
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                permutation = torch.randperm(len(y), device=device)
                f = lam * f + (1.0 - lam) * f[permutation]
                raw = lam * raw + (1.0 - lam) * raw[permutation]
                y_second = y[permutation]
            output, _ = model(f, raw, subject)
            loss = nn.functional.cross_entropy(
                output, y, weight=weights, label_smoothing=args.label_smoothing
            )
            if args.mixup_alpha > 0:
                second = nn.functional.cross_entropy(
                    output, y_second, weight=weights,
                    label_smoothing=args.label_smoothing
                )
                loss = lam * loss + (1.0 - lam) * second
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step(); losses.append(loss.item())
        scheduler.step()
        val = evaluate(model, loaders["val"], device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **val}
        history.append(row)
        print(json.dumps({k: v for k, v in row.items()
                          if k not in ("confusion_matrix", "subject_accuracy")}), flush=True)
        selection_value = val[args.selection_metric]
        if selection_value > best:
            best, stale = selection_value, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args),
                        "validation": val}, run / "best.pt")
        else:
            stale += 1
        if stale >= args.patience:
            break
    ckpt = torch.load(run / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    serializable_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    result = {"best_epoch": ckpt["epoch"], "args": serializable_args,
              "validation": evaluate(model, loaders["val"], device),
              "test": evaluate(model, loaders["test"], device), "history": history}
    (run / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL " + json.dumps({k: v for k, v in result.items()
                                 if k not in ("history", "args")}), flush=True)


if __name__ == "__main__":
    main()
