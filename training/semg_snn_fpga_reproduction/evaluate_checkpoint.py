from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from model import PaperSNN, PaperSNNWithDelays
from train import CLASS_NAMES, NPZDataset, evaluate


def quantize_weights_int8(model: PaperSNN) -> PaperSNN:
    """Symmetric per-tensor fake INT8 quantization."""
    quantized = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in quantized.parameters():
            maximum = parameter.abs().max()
            if maximum == 0:
                continue
            scale = maximum / 127.0
            parameter.copy_(torch.round(parameter / scale).clamp(-127, 127) * scale)
    return quantized


def two_window_voting(
    predictions: list[int], subjects: list[int], starts: list[int]
) -> list[int]:
    """Debounce isolated predictions using two consecutive matching windows.

    The paper says that non-consecutive labels are discarded and the last valid
    prediction is restored, but it does not publish executable pseudocode.
    This implementation accepts a class transition after two consecutive windows
    predict the same candidate class, independently for each subject.
    """
    output = list(predictions)
    subject_array = np.asarray(subjects)
    start_array = np.asarray(starts)
    prediction_array = np.asarray(predictions)

    for subject in np.unique(subject_array):
        indexes = np.flatnonzero(subject_array == subject)
        indexes = indexes[np.argsort(start_array[indexes])]
        if len(indexes) == 0:
            continue
        accepted = int(prediction_array[indexes[0]])
        candidate = accepted
        candidate_count = 0
        output[indexes[0]] = accepted

        for index in indexes[1:]:
            prediction = int(prediction_array[index])
            if prediction == accepted:
                candidate = accepted
                candidate_count = 0
            elif prediction == candidate:
                candidate_count += 1
                if candidate_count >= 2:
                    accepted = candidate
                    candidate_count = 0
            else:
                candidate = prediction
                candidate_count = 1
            output[index] = accepted
    return output


def compact_metrics(targets: list[int], predictions: list[int]) -> dict:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=list(range(13))
        ).tolist(),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=root / "runs" / "full_baseline" / "best.pt"
    )
    parser.add_argument(
        "--data", type=Path, default=root / "data" / "processed" / "test.npz"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "runs" / "full_baseline" / "test_metrics.json",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = state["args"]
    max_delay = int(saved_args.get("max_delay", 0))
    if max_delay > 0:
        model = PaperSNNWithDelays(
            decay=float(saved_args["decay"]),
            threshold=float(saved_args["threshold"]),
            max_delay=max_delay,
            initial_delay=float(saved_args.get("initial_delay", 1.0)),
        ).to(device)
    else:
        model = PaperSNN(
            decay=float(saved_args["decay"]),
            threshold=float(saved_args["threshold"]),
        ).to(device)
    model.load_state_dict(state["model"])

    dataset = NPZDataset(args.data)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    raw = evaluate(model, loader, device)
    raw_voted = two_window_voting(
        raw["predictions"], raw["subjects"], raw["starts"]
    )

    quantized_model = quantize_weights_int8(model).to(device)
    quantized = evaluate(quantized_model, loader, device)
    quantized_voted = two_window_voting(
        quantized["predictions"], quantized["subjects"], quantized["starts"]
    )

    result = {
        "checkpoint_epoch": int(state["epoch"]),
        "device": str(device),
        "class_names": CLASS_NAMES,
        "fp32_raw": raw,
        "fp32_two_window_vote": compact_metrics(raw["targets"], raw_voted),
        "int8_fake_quantized_raw": quantized,
        "int8_fake_quantized_two_window_vote": compact_metrics(
            quantized["targets"], quantized_voted
        ),
        "voting_assumption": (
            "A transition is accepted after two consecutive windows predict the "
            "same candidate. The paper does not provide exact voting pseudocode."
        ),
        "quantization_assumption": (
            "Symmetric per-layer fake INT8 weight quantization; activations and "
            "LIF states remain floating point."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "checkpoint_epoch": state["epoch"],
                "fp32_raw": compact_metrics(
                    raw["targets"], raw["predictions"]
                ) | {
                    "sparsity": raw["sparsity"],
                    "layer_spike_rates": raw["layer_spike_rates"],
                },
                "fp32_voted": compact_metrics(raw["targets"], raw_voted),
                "int8_raw": compact_metrics(
                    quantized["targets"], quantized["predictions"]
                ),
                "int8_voted": compact_metrics(
                    quantized["targets"], quantized_voted
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
