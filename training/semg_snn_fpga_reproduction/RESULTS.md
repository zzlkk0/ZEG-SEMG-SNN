# Reproduction Results

Run date: 2026-07-28

## Outcome

The NinaPro DB5 preprocessing protocol was reproduced exactly enough to generate
the same number of test windows as the paper:

| Split | Windows |
|---|---:|
| Train | 44,630 |
| Validation | 11,060 |
| Test | 11,276 |

The paper's published confusion matrix also contains 11,276 test windows. This
strongly validates the implementation of the E1 class selection, repetition
split, initial delay, window length, shift, and label-purity rules.

## Training setup

- GPU: NVIDIA GeForce RTX 3080, 20 GB
- PyTorch: 2.5.1
- CUDA build: 12.4
- Existing environment: `Python environment`
- Training epochs completed: 30
- Selected checkpoint: epoch 28, based on validation loss
- Parameters: 23,360
- Batch size: 32
- Optimizer: Adam
- Learning rate: `1e-3`
- Network: `96 → 64 → 128 → 64 → 13`
- LIF decay: 0.9
- LIF threshold: 1.0
- Rate targets: true 0.2, false 0.03

## Test results

| Model/evaluation | Accuracy | Macro-F1 |
|---|---:|---:|
| FP32, raw windows | **83.3895%** | 0.6322 |
| FP32, assumed two-window voting | **83.4161%** | 0.6426 |
| Fake INT8 weights, raw windows | **83.1146%** | 0.6310 |
| Fake INT8 weights, assumed two-window voting | **83.1146%** | 0.6430 |
| Paper result | **85.6%** | Not reported |

The FP32 raw result is approximately 2.21 percentage points below the paper.
The fake INT8 weight quantization changes accuracy by only -0.27 percentage
points, consistent with the paper's statement that 8-bit weights cause a
negligible drop.

## Learnable axonal-delay experiment

A second experiment approximated SLAYER's `delay=True` behavior with one
learnable axonal delay per output neuron on each of the first three layers. The
delays were constrained to 0–62 time steps and trained through linear temporal
interpolation. Training started from the epoch-28 no-delay checkpoint and ran for
15 additional epochs.

| Model/evaluation | Accuracy | Macro-F1 |
|---|---:|---:|
| No-delay FP32, raw windows | 83.3895% | 0.6322 |
| **Learnable-delay FP32, raw windows** | **83.9305%** | **0.6561** |
| Learnable-delay FP32, assumed voting | 83.6201% | 0.6611 |
| Learnable-delay fake INT8, raw windows | 83.4871% | 0.6423 |
| Learnable-delay fake INT8, assumed voting | 83.3097% | 0.6498 |

The learnable-delay model improves raw test accuracy by approximately **0.54
percentage points** and Macro-F1 by **0.024**. Its test sparsity is 92.50%.

At the final training epoch, learned delay statistics were:

| Delayed layer | Minimum | Mean | Maximum |
|---|---:|---:|---:|
| Layer 1 | 0.74 | 1.67 | 2.67 |
| Layer 2 | 0.42 | 1.65 | 8.11 |
| Layer 3 | 0.35 | 4.95 | 33.47 |

The assumed two-window voting rule slightly reduces accuracy for this model.
This confirms that the paper's unpublished voting details should not be guessed
and presented as an exact reproduction. Raw window accuracy remains the primary
metric.

Artifacts:

- Delay checkpoint: `runs/delay62_finetune/best.pt`
- Detailed delay metrics: `runs/delay62_finetune/test_metrics.json`

## Spike activity

FP32 test-set layer spike rates:

| Layer | Spike rate |
|---|---:|
| Dense LIF 1 | 12.2356% |
| Dense LIF 2 | 5.8201% |
| Dense LIF 3 | 6.8068% |
| Output LIF | 3.8423% |

Mean sparsity across layers:

```text
92.8238%
```

Paper-reported software sparsity:

```text
90.99%
```

The reproduction is slightly more sparse, which may partially explain the lower
classification accuracy.

## Validation progression

Selected checkpoints during training:

| Epoch | Validation accuracy | Macro-F1 |
|---:|---:|---:|
| 1 | 71.51% | 0.344 |
| 3 | 75.73% | 0.475 |
| 7 | 79.41% | 0.556 |
| 12 | 81.34% | 0.616 |
| 18 | 82.97% | 0.656 |
| 24 | 83.36% | 0.668 |
| 28 | 83.52% | 0.666 |
| 30 | 83.36% | 0.669 |

Training was manually stopped after epoch 30 because the validation result had
entered a plateau. The best checkpoint had already been saved.

## Differences from the paper

The following details are either absent from the paper or not reproduced in the
initial implementation:

1. The paper uses SLAYER/Lava; this reproduction uses a pure PyTorch
   surrogate-gradient implementation.
2. The paper enables axonal delays up to 62 time steps in the first three dense
   layers. This baseline does not yet implement trainable per-synapse delays.
3. LIF decay and voltage threshold are not reported by the paper.
4. The exact derivative operator is not reported. This implementation uses
   `numpy.gradient` twice.
5. The exact SLAYER surrogate gradient and initialization are not reported.
6. The temporal voting algorithm is described but no executable pseudocode is
   supplied. The reported voting result here requires two consecutive candidate
   windows before accepting a class transition.
7. INT8 results use symmetric fake weight quantization. LIF states and
   activations remain floating point.

## Interpretation

The software-side method is successfully reproduced:

- the data split matches the published test-set size exactly;
- the network trains stably on the existing CUDA environment;
- raw accuracy reaches 83.39%, close to the published 85.6%;
- fake INT8 quantization has a small effect;
- the network exhibits high spike sparsity.

The delay experiment confirms that temporal alignment is useful, but a gap of
approximately 1.67 percentage points remains relative to the paper. The next
steps are an exact Lava-DL/SLAYER run and a controlled sweep over LIF decay and
threshold, because the paper does not disclose those values.

## Reproduction artifacts

- Dataset metadata: `data/processed/metadata.json`
- Best checkpoint: `runs/full_baseline/best.pt`
- Detailed metrics: `runs/full_baseline/test_metrics.json`
- Dataset preparation: `prepare_db5.py`
- Model: `model.py`
- Training: `train.py`
- Evaluation and quantization: `evaluate_checkpoint.py`
