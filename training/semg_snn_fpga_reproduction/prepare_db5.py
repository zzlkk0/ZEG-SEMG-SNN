from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


TRAIN_REPS = {1, 2, 4, 6}
VAL_REPS = {3}
TEST_REPS = {5}


def delta_modulate(signal: np.ndarray, threshold: float) -> np.ndarray:
    """Paper Algorithm 1, independently for every channel.

    Args:
        signal: float array [time, channels].
        threshold: positive delta threshold.

    Returns:
        Boolean array [time, channels * 2], ordered POS then NEG.
    """
    time, channels = signal.shape
    pos = np.zeros((time, channels), dtype=np.bool_)
    neg = np.zeros((time, channels), dtype=np.bool_)
    reference = signal[0].astype(np.float32, copy=True)

    for t in range(1, time):
        sample = signal[t]
        positive = sample > reference + threshold
        negative = sample < reference - threshold
        pos[t] = positive
        neg[t] = negative
        update = positive | negative
        reference[update] = sample[update]

    return np.concatenate((pos, neg), axis=1)


def encode_emg(emg: np.ndarray, threshold: float) -> np.ndarray:
    emg = np.asarray(emg, dtype=np.float32)
    first = np.gradient(emg, axis=0)
    second = np.gradient(first, axis=0)
    traces = [
        delta_modulate(emg, threshold),
        delta_modulate(first, threshold),
        delta_modulate(second, threshold),
    ]
    encoded = np.concatenate(traces, axis=1)
    if encoded.shape[1] != 96:
        raise ValueError(f"Expected 96 encoded channels, got {encoded.shape}")
    return encoded


def choose_split(rep_window: np.ndarray) -> str | None:
    nonzero = rep_window[rep_window > 0]
    if len(nonzero) == 0:
        return None
    values, counts = np.unique(nonzero, return_counts=True)
    rep = int(values[np.argmax(counts)])
    if rep in TRAIN_REPS:
        return "train"
    if rep in VAL_REPS:
        return "val"
    if rep in TEST_REPS:
        return "test"
    return None


def label_window(labels: np.ndarray, exercise_purity: float) -> int | None:
    if np.all(labels == 0):
        return 0
    nonzero = labels[labels > 0]
    if len(nonzero) == 0:
        return None
    values, counts = np.unique(nonzero, return_counts=True)
    index = int(np.argmax(counts))
    label = int(values[index])
    if counts[index] / len(labels) >= exercise_purity:
        return label
    return None


def process_subject(
    mat_path: Path,
    threshold: float,
    window: int,
    shift: int,
    initial_delay: int,
    exercise_purity: float,
) -> dict[str, list]:
    mat = loadmat(mat_path)
    emg = np.asarray(mat["emg"], dtype=np.float32)
    labels = np.asarray(mat["restimulus"]).reshape(-1).astype(np.int16)
    reps = np.asarray(mat["rerepetition"]).reshape(-1).astype(np.int16)
    subject = int(np.asarray(mat.get("subject", [[0]])).reshape(-1)[0])

    if emg.shape[1] != 16:
        raise ValueError(f"{mat_path}: expected 16 EMG channels, got {emg.shape}")
    if not (len(emg) == len(labels) == len(reps)):
        raise ValueError(f"{mat_path}: inconsistent lengths")

    encoded = encode_emg(emg, threshold)
    out = {split: [] for split in ("train", "val", "test")}

    for start in range(initial_delay, len(emg) - window + 1, shift):
        stop = start + window
        label = label_window(labels[start:stop], exercise_purity)
        if label is None:
            continue
        split = choose_split(reps[start:stop])
        if split is None:
            continue
        out[split].append(
            (
                encoded[start:stop],
                label,
                subject,
                start,
                int(np.max(reps[start:stop])),
            )
        )
    return out


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, default=root / "data" / "extracted"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "data" / "processed"
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--threshold", type=float, default=15.0)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--shift", type=int, default=20)
    parser.add_argument("--initial-delay", type=int, default=400)
    parser.add_argument("--exercise-purity", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = {split: [] for split in ("train", "val", "test")}

    for subject in args.subjects:
        mat_path = args.input_dir / f"s{subject}" / f"S{subject}_E1_A1.mat"
        if not mat_path.exists():
            raise FileNotFoundError(mat_path)
        result = process_subject(
            mat_path,
            threshold=args.threshold,
            window=args.window,
            shift=args.shift,
            initial_delay=args.initial_delay,
            exercise_purity=args.exercise_purity,
        )
        counts = {split: len(rows) for split, rows in result.items()}
        print(f"S{subject}: {counts}", flush=True)
        for split in combined:
            combined[split].extend(result[split])

    summary: dict[str, dict] = {}
    for split, rows in combined.items():
        if not rows:
            raise RuntimeError(f"No rows for {split}")
        x = np.stack([row[0] for row in rows]).astype(np.bool_)
        y = np.asarray([row[1] for row in rows], dtype=np.int16)
        subjects = np.asarray([row[2] for row in rows], dtype=np.int8)
        starts = np.asarray([row[3] for row in rows], dtype=np.int32)
        repetitions = np.asarray([row[4] for row in rows], dtype=np.int8)
        destination = args.output_dir / f"{split}.npz"
        np.savez_compressed(
            destination,
            x=x,
            y=y,
            subject=subjects,
            start=starts,
            repetition=repetitions,
        )
        labels, counts = np.unique(y, return_counts=True)
        summary[split] = {
            "samples": int(len(y)),
            "shape": list(x.shape),
            "label_counts": {
                str(int(label)): int(count)
                for label, count in zip(labels, counts, strict=True)
            },
            "file": str(destination),
        }
        print(split, summary[split], flush=True)

    metadata = {
        "protocol": {
            "threshold": args.threshold,
            "window": args.window,
            "shift": args.shift,
            "initial_delay": args.initial_delay,
            "exercise_purity": args.exercise_purity,
            "train_repetitions": sorted(TRAIN_REPS),
            "validation_repetitions": sorted(VAL_REPS),
            "test_repetitions": sorted(TEST_REPS),
            "derivative": "numpy.gradient",
            "encoding_order": [
                "raw_positive",
                "raw_negative",
                "first_positive",
                "first_negative",
                "second_positive",
                "second_negative",
            ],
        },
        "splits": summary,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
