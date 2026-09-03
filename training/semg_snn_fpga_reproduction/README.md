# NinaPro DB5 + Low-Power FPGA SNN Reproduction

This project reproduces the software-side method from:

> M. A. Scrugli, G. Leone, P. Busia, P. Meloni, “sEMG-Based Gesture
> Recognition with Spiking Neural Networks on Low-Power FPGA,” DASIP 2024.
> DOI: 10.1007/978-3-031-62874-0_2.

The accepted manuscript is available from the
[institutional repository](https://iris.unica.it/handle/11584/469172).
The repository rejected direct command-line PDF download with HTTP 403, but the
paper was inspected through its open institutional viewer.

## Reproduced protocol

- Dataset: NinaPro DB5, exercise E1.
- Classes: rest plus 12 finger movements (13 classes).
- Input: all 16 Myo EMG channels.
- Features: raw EMG, first derivative, second derivative.
- Encoding: delta modulation with threshold 15, positive and negative traces.
- Input dimensions: `16 channels × 3 signal orders × 2 polarities = 96`.
- Window: 0.5 s = 100 samples.
- Shift: 0.1 s = 20 samples.
- Initial discarded interval: 2 s = 400 samples.
- Exercise window: at least 80% of samples have the same non-rest label.
- Rest window: 100% of samples have label 0.
- Training repetitions: 1, 2, 4, 6.
- Validation repetition: 3.
- Test repetition: 5.
- Network: dense `96 → 64 → 128 → 64 → 13`, LIF neurons.
- Optimizer: Adam, learning rate `1e-3`, batch size 32.
- Loss: output spike-rate targets 0.2 for the true class and 0.03 otherwise.
- Early stopping patience: 10 epochs.

## Known reproduction gaps

The paper does not report the LIF decay, voltage threshold, surrogate-gradient
parameters, exact derivative implementation, learned delay state, random seed,
number of epochs before early stopping, or SLAYER weight initialization.

This implementation uses:

- pure PyTorch surrogate-gradient LIF neurons;
- decay `0.9`, threshold `1.0`, reset-to-zero;
- `numpy.gradient` for first and second derivatives;
- fast-sigmoid surrogate gradient;
- no trainable axonal delays in the initial baseline.

The paper used SLAYER/Lava and enabled axonal delays up to 62 time steps in
the first three dense layers. Therefore, matching 85.6% exactly is not expected
until the unspecified settings and delay implementation are resolved.

## Existing environment

The project reuses:

```text
Python environment
```

No separate environment is required.

## Commands

Extract DB5:

```bash
bash scripts/extract_db5.sh
```

Prepare paper-compatible windows:

```bash
python prepare_db5.py
```

Run a short smoke test:

```bash
python train.py \
  --epochs 2 \
  --max-train-batches 20 \
  --max-eval-batches 10
```

Run full training:

```bash
python train.py --epochs 100
```

Fine-tune the baseline with the paper's maximum axonal delay:

```bash
python train.py \
  --epochs 30 \
  --max-delay 62 \
  --initial-delay 1 \
  --init-checkpoint runs/full_baseline/best.pt \
  --run-dir runs/delay62_finetune
```

Evaluate a saved checkpoint, including fake INT8 weight quantization and the
documented two-window voting assumption:

```bash
python evaluate_checkpoint.py
```

GPU access must be available to the process. The script automatically uses CUDA
when PyTorch can see it and otherwise falls back to CPU.

## Outputs

```text
data/raw/          downloaded subject ZIP files
data/extracted/    extracted MATLAB files
data/processed/    prepared NPZ splits
runs/              metrics, checkpoints, confusion matrices
```

## Evaluation

The script reports:

- raw per-window accuracy;
- macro-F1;
- confusion matrix;
- output and hidden-layer spike rates;
- overall spike sparsity.

The paper's temporal voting filter is implemented as an optional evaluation
step. Its exact behavior is not fully specified, so raw window accuracy remains
the primary reproducible metric.

See [`RESULTS.md`](RESULTS.md) for the completed baseline run.

The completed learnable-delay approximation is stored in
`runs/delay62_finetune`; it improves raw test accuracy from 83.39% to 83.93%.
