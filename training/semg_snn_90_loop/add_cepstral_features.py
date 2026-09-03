from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.fft import dct


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data"
OUT = ROOT / "data_cepstral"


def cepstral(raw):
    spectrum = np.abs(np.fft.rfft(raw.astype(np.float32), axis=1)) ** 2
    log_spectrum = np.log1p(spectrum)
    coefficients = dct(log_spectrum, type=2, axis=1, norm="ortho")[:, 1:9]
    return coefficients.reshape(len(raw), -1).astype(np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for split in ("train", "val", "test"):
        source = np.load(SOURCE / f"{split}.npz")
        values = {key: source[key] for key in source.files}
        values["features"] = np.concatenate(
            (values["features"], cepstral(values["raw"])), axis=1
        )
        arrays[split] = values
        np.savez_compressed(OUT / f"{split}.npz", **values)
        print(split, values["features"].shape, flush=True)
    train = arrays["train"]
    mean = np.zeros((10, train["features"].shape[1]), np.float32)
    std = np.ones_like(mean)
    for subject in range(10):
        mask = train["subject"] == subject
        mean[subject] = train["features"][mask].mean(axis=0)
        std[subject] = train["features"][mask].std(axis=0) + 1e-4
    old = np.load(SOURCE / "normalization.npz")
    np.savez(OUT / "normalization.npz", feature_mean=mean, feature_std=std,
             raw_mean=old["raw_mean"], raw_std=old["raw_std"])
    (OUT / "metadata.json").write_text(json.dumps({
        "feature_count": train["features"].shape[1],
        "added": "8 DCT coefficients of per-channel log-power spectrum",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
