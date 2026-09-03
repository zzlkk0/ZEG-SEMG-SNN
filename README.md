# ZEG-SEMG-SNN

Research and teaching code for 13-class surface electromyography (sEMG)
gesture recognition on NinaPro DB5 using spiking neural networks (SNNs).

The repository covers the full progression from a baseline Delay-SNN
reproduction through Context/Hybrid multi-branch optimization, SNN-LSTM and
RF-SNN comparisons, hardware-friendly quantization-aware training (QAT),
fixed-point reference inference, and FPGA deployment tutorials.

> This repository does not contain NinaPro data, trained checkpoints, exported
> weights, or bitstreams. Accuracy values are software results under a specific
> split unless physical-board evidence is explicitly provided.

## Main results

The main experiments use NinaPro DB5 Exercise 1: Rest plus 12 gestures,
16 channels at 200 Hz, 500 ms windows, and a 100 ms hop. Repetitions 1/2/4/6
are used for training, repetition 3 for validation, and repetition 5 for test.

| Route | Test accuracy | Interpretation |
|---|---:|---|
| Delay-SNN | 83.93% | Pure PyTorch SNN with learnable axonal delays |
| Best SNN-LSTM experiment | 83.84% | Did not outperform Delay-SNN |
| RF-SNN, strict split | 80.24% | 52-sample RF80 experiment |
| Context + Hybrid + Delay | **91.10%** | FP32 software result without gesture boundaries or future information |
| Hardware-friendly QAT ensemble | **91.11%** | Fixed-point/LUT software reference; not an RTL or FPGA measurement |

This is a cross-repetition evaluation on the same subjects, not
leave-one-subject-out evaluation. Windows overlap by 80%. Read each experiment's
`RESULTS.md` before citing a number outside its protocol.

## Repository layout

```text
docs/
  tutorial/                     Eight English chapters and notebooks, from SNN basics to FPGA deployment
  specifications/               Hardware-friendly retraining and Delay branch interface specifications
training/
  semg_snn_fpga_reproduction/   Software reproduction of the Delay-SNN method
  semg_snn_90_loop/             Context/Hybrid training, ensemble evaluation, QAT, and fixed-point export
  semg_snn_lstm/                SNN-LSTM comparison experiments
  semg_rf_snn/                  RF-SNN comparison experiments
scripts/
  sanitize_notebooks.py         Removes notebook outputs and workstation paths
```

## Quick start

Python 3.10–3.12 is recommended. CUDA installation depends on the host driver;
for GPU use, install an appropriate PyTorch build first and then install the
remaining dependencies.

```bash
git clone https://github.com/zzlkk0/ZEG-SEMG-SNN.git
cd ZEG-SEMG-SNN

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run checks that do not require the dataset:

```bash
python -m compileall -q training
python scripts/check_public_release.py
```

Start with the tutorials:

```bash
python -m pip install -r requirements-notebooks.txt
jupyter lab
```

Open [`docs/tutorial/00-README.md`](docs/tutorial/00-README.md). The synthetic
examples in Chapters 01–08 are self-contained. The real-project notebooks in
`docs/tutorial/notebooks/` require NinaPro DB5 data and checkpoints as described
in [`DATASET.md`](DATASET.md).

## Training entry points

Baseline Delay-SNN:

```bash
cd training/semg_snn_fpga_reproduction
python prepare_db5.py --input-dir data/extracted --output-dir data/processed
python train.py --epochs 100 --run-dir runs/baseline
```

Context/Hybrid main route:

```bash
cd training/semg_snn_90_loop
python prepare.py
python prepare_continuous_context.py

python train.py \
  --model context_class_adaptive \
  --context 23 \
  --continuous-context \
  --stream-context \
  --epochs 24 \
  --patience 7 \
  --batch-size 256 \
  --lr 0.0002 \
  --run-name context23_class_plif_stream
```

Hardware-friendly QAT:

```bash
cd training/semg_snn_90_loop
python train_qat.py --help
python export_hw_fixed.py --help
python evaluate_hw_ensemble.py --help
```

Each subproject README documents its initialization dependencies, complete
training commands, and evaluation procedure.

## Reproducibility rules

- Compute normalization statistics from training repetitions only.
- Select checkpoints, ensemble weights, and the Rest bias on validation data.
- Use the test split for final reporting, not hyperparameter tuning.
- Report accuracy, macro-F1, and non-Rest gesture accuracy together.
- Distinguish FP32, QAT/fixed-point reference, HLS/RTL simulation, and physical
  FPGA measurements.
- Use a new `runs/<name>` directory for each experiment.

## Data and generated models

NinaPro data and training artifacts are intentionally excluded. See
[`DATASET.md`](DATASET.md) for the expected layout and preparation steps.
`.gitignore` blocks common datasets, checkpoints, exported weights, bitstreams,
and tool-generated build directories.

## References and limitations

The Delay-SNN reproduction is based on:

> M. A. Scrugli, G. Leone, P. Busia, P. Meloni,
> “sEMG-Based Gesture Recognition with Spiking Neural Networks on Low-Power FPGA,”
> DASIP 2024. DOI: 10.1007/978-3-031-62874-0_2.

See [`training/semg_rf_snn/RESEARCH.md`](training/semg_rf_snn/RESEARCH.md) for
the RF-SNN background and known reproduction differences.

No license is included yet. Do not assume permission for commercial use until
a license is explicitly added.
