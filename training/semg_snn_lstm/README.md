# NinaPro DB5 SNN-LSTM

Independent hybrid-model project derived from the completed low-power SNN
reproduction.

## Goal

Improve the previous raw test accuracy of **83.93%** by combining:

```text
Delta-encoded sEMG
→ 96→64 LIF
→ 64→128 LIF
→ reset-before membrane sequence
→ LayerNorm
→ one-layer unidirectional LSTM(64)
→ temporal mean pooling
→ 13-class linear head
```

The LSTM receives the second LIF layer's pre-reset membrane potential. This
preserves sub-threshold information that is discarded by a spike-only
interface.

## Project layout

- Existing SNN project: `../semg_snn_fpga_reproduction`
- This project: `../semg_snn_lstm`
- Dataset: relative symlink to the existing project's `data` directory
- Existing environment: `Python environment`

No data is duplicated and no new Conda environment is created.

## Training

```bash
python train.py \
  --feature-mode membrane \
  --epochs 40 \
  --run-dir runs/membrane
```

Alternative interfaces:

```bash
# Feed binary spikes to the LSTM
--feature-mode spike

# Concatenate membrane and spike features
--feature-mode both
```

The first two SNN layers are initialized from the previous project's best
no-delay checkpoint, then jointly fine-tuned with the LSTM.

Residual model that preserves the complete pretrained SNN classifier and learns
an LSTM correction:

```bash
python train.py \
  --architecture residual \
  --freeze-snn \
  --epochs 30 \
  --run-dir runs/residual_frozen
```

The residual head is initialized to zero, so before training its class ranking is
identical to the pretrained pure SNN. This avoids damaging the 83.39% no-delay
baseline while the LSTM learns corrections.

Residual correction on top of the stronger 83.93% learned-delay SNN:

```bash
python train.py \
  --architecture delayed_residual \
  --freeze-snn \
  --frontend-checkpoint \
    ../semg_snn_fpga_reproduction/runs/delay62_finetune/best.pt \
  --epochs 25 \
  --run-dir runs/delayed_residual_frozen
```

## Important interpretation

This is a hybrid ANN-SNN model. It may improve recognition accuracy, but the
LSTM adds dense multiply-accumulate operations and sigmoid/tanh gates. Its power
consumption cannot be compared directly with the approximately 1.7 mW FPGA SNN
reported by the DASIP paper.

See [`RESULTS.md`](RESULTS.md) for all three completed experiments. The best
SNN-LSTM test accuracy is 83.84%; it does not exceed the 83.93% learned-delay
pure SNN.
