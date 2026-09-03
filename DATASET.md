# Data and Model Files

## Why data and checkpoints are excluded

NinaPro DB5 is subject to the dataset publisher's terms, and the files are too
large for ordinary Git history. Trained checkpoints, processed windows,
exported weights, and bitstreams are reproducible artifacts, so this repository
contains source code, documentation, and aggregate results only.

Obtain DB5 through the official NinaPro distribution channel and comply with
its citation and usage requirements. Do not commit raw participant data or a
repackaged copy of the dataset to this repository.

## Expected layout

```text
training/semg_snn_fpga_reproduction/
  data/
    extracted/        Extracted DB5 MATLAB files
    processed/        train/val/test.npz produced by prepare_db5.py
  runs/               Checkpoints and metrics

training/semg_snn_90_loop/
  data/               Features produced by prepare.py
  runs/               Context, Hybrid, and QAT checkpoints/metrics
  weights_hw/         Fixed-point exports and golden vectors
```

These directories are excluded by `.gitignore`.

## Fixed evaluation split

| Split | Repetitions | Windows |
|---|---|---:|
| Train | 1, 2, 4, 6 | 44,630 |
| Validation | 3 | 11,060 |
| Test | 5 | 11,276 |

Windows contain 100 samples with a shift of 20 samples and begin after sample
400 of each recording. Gesture windows require at least 80% label agreement;
Rest windows must contain label 0 throughout.

## Data preparation

```bash
cd training/semg_snn_fpga_reproduction
python prepare_db5.py \
  --input-dir data/extracted \
  --output-dir data/processed

cd ../semg_snn_90_loop
python prepare.py
python prepare_continuous_context.py
```

Some optimization scripts read the adjacent
`semg_snn_fpga_reproduction/data/processed` directory. Keep the default
repository layout and no source changes should be required.

## Privacy requirements

When using newly collected sEMG data:

- Do not commit raw participant recordings, names, identifier maps, consent
  forms, or device serial numbers.
- Confirm the ethics approval and participant authorization before publishing
  statistics.
- Prefer synthetic data or explicitly authorized anonymized samples for public
  examples.
- Do not record local absolute paths, usernames, or access tokens in logs.
