# SNN-LSTM Experiment Results

Run date: 2026-07-28

## Objective

Improve the previous learned-delay pure-SNN test accuracy of **83.93%** by
adding a one-layer, unidirectional LSTM to the NinaPro DB5 E1 classifier.

All experiments use the exact same processed splits:

| Split | Windows |
|---|---:|
| Train | 44,630 |
| Validation | 11,060 |
| Test | 11,276 |

## Environment

- Existing Conda environment: `Python environment`
- GPU: NVIDIA GeForce RTX 3080, 20 GB
- PyTorch: 2.5.1
- CUDA build: 12.4

No new Conda environment was created.

## Experiment 1: Direct membrane-to-LSTM

Architecture:

```text
96 input spikes
→ 64 LIF
→ 128 LIF
→ pre-reset membrane sequence
→ LayerNorm
→ unidirectional LSTM(64)
→ temporal mean
→ 13 classes
```

The first two LIF layers were initialized from the previous pure-SNN
checkpoint and jointly fine-tuned.

| Metric | Result |
|---|---:|
| Best epoch | 8 |
| Validation accuracy | 82.05% |
| Test accuracy | **82.57%** |
| Test Macro-F1 | 0.6221 |
| Parameters | 65,101 |
| LIF sparsity | 92.92% |

Interpretation: joint fine-tuning damaged the already useful SNN representation.
Training loss continued to decrease while validation performance plateaued,
showing clear overfitting.

Artifacts:

- `runs/membrane/best.pt`
- `runs/membrane/metrics.json`

## Experiment 2: Frozen SNN plus LSTM residual

Architecture:

```text
Frozen pretrained 4-layer no-delay SNN → base spike-rate logits
                         +
Layer-2 membrane → LSTM → zero-initialized residual logits
```

This model starts with the same class ranking as the pretrained pure SNN. The
SNN is frozen and only the LSTM correction branch is trained.

| Metric | Result |
|---|---:|
| Best epoch | 27 |
| Validation accuracy | 84.28% |
| Test accuracy | **83.84%** |
| Test Macro-F1 | 0.6531 |
| Parameters | 74,125 |
| Trainable parameters | 50,765 |

This is the best SNN-LSTM result. It is:

- 1.27 points better than the direct hybrid;
- about 0.45 points better than the no-delay pure SNN;
- about **0.09 points below** the learned-delay pure SNN.

Artifacts:

- `runs/residual_frozen/best.pt`
- `runs/residual_frozen/metrics.json`

## Experiment 3: Frozen learned-delay SNN plus LSTM residual

Architecture:

```text
Frozen learned-delay SNN → base spike-rate logits
                  +
Layer-2 membrane → LSTM → zero-initialized residual logits
```

The frozen base is the previous 83.93% learned-delay SNN.

| Metric | Result |
|---|---:|
| Best epoch | 8 |
| Validation accuracy | 84.14% |
| Test accuracy | **83.70%** |
| Test Macro-F1 | 0.6486 |
| Parameters | 74,381 |
| Trainable parameters | 50,765 |

The residual branch slightly overfits the validation subjects/repetitions and
does not improve the independent test repetition.

Artifacts:

- `runs/delayed_residual_frozen/best.pt`
- `runs/delayed_residual_frozen/metrics.json`

## Final comparison

| Model | Test accuracy | Macro-F1 |
|---|---:|---:|
| Pure SNN, no delay | 83.39% | 0.6322 |
| **Pure SNN, learned delay** | **83.93%** | **0.6561** |
| Direct SNN-LSTM | 82.57% | 0.6221 |
| Frozen SNN + LSTM residual | 83.84% | 0.6531 |
| Delayed SNN + LSTM residual | 83.70% | 0.6486 |

The target of exceeding 83.93% was **not achieved**. The best SNN-LSTM reaches
83.84%, only 0.09 percentage points below the best pure SNN.

## Why LSTM did not improve this task

Likely reasons:

1. Each sample is already a short 0.5-second isolated-gesture window.
2. The SNN membrane dynamics and learned delays already model short-term
   temporal dependencies.
3. The LSTM adds approximately 50k trainable parameters, increasing overfitting.
4. The training and test repetitions come from the same subjects, so longer-term
   sequence modeling has limited additional information.
5. Mean pooling can dilute brief discriminative transitions.
6. The dominant rest class encourages conservative temporal predictions.

LSTM is more likely to help on:

- longer windows;
- complete gesture repetitions instead of 0.5-second windows;
- continuous sign-language sequences;
- variable-duration gestures;
- multi-window online decoding.

## Recommended next experiment

Do not increase the LSTM size. The next accuracy-oriented experiment should use
one of:

1. temporal attention instead of mean pooling;
2. complete 5-second repetitions with a hierarchical window encoder;
3. a small TCN/GRU residual branch;
4. ensemble calibration between the learned-delay SNN and an independently
   trained CNN/TCN;
5. class-balanced fine-tuning evaluated primarily by Macro-F1.

For low-power deployment, the learned-delay pure SNN remains the preferred
model.
