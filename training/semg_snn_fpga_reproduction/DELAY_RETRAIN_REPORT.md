# Delay-SNN Retraining Report: Paper Alignment and Sparsity Regularization

Related specification: `../../docs/specifications/DELAY_RETRAIN_TOWARD_PAPER_SPEC.md`.

This report covers only the Delay-SNN branch in `semg_snn_fpga_reproduction`. It does not cover Context/Hybrid or hardware power and duty-cycle implementation.

## 1. Executive summary

| Item | Result |
|---|---|
| paper training recipe | existing recipe already matches all published settings |
| window-label purity | existing preprocessing already matches the paper |
| temporal voting | `paper_style_vote` improves 83.9305% to **84.3828%** (+0.4527 pp); the causal control loses accuracy |
| sparsity sweep | all six lambda settings converged; `lambda=0.001` is the most useful trade-off |
| fixed-point compatibility | baseline and `lambda=0.001/0.002` export through the unchanged deployment tool and retain the existing format |

## 2. Training-recipe and labeling audit

The recorded checkpoint arguments and source were compared with the published settings:

| Hyperparameter | Paper description | Existing project | Match |
|---|---|---|:---:|
| optimizer | Adam | `torch.optim.Adam` | yes |
| learning rate | 0.001 | 0.001 | yes |
| batch size | 32 | 32 | yes |
| early stopping | 10 epochs without improvement | `patience=10` | yes |
| loss | SpikeRate targets 0.2 true / 0.03 false | `spike_rate_loss(0.2, 0.03)` | yes |

Consequently, the existing 83.9305% checkpoint already uses the published recipe; no separate recipe-alignment retraining was justified.

Unpublished details remain genuine reproduction uncertainties: LIF `decay=0.9`, `threshold=1.0`, surrogate slope 10, plain PyTorch instead of SLAYER/Lava, and differentiable linear interpolation instead of SLAYER's native delay operator.

`prepare_db5.py::label_window()` was also audited. It accepts a gesture only when that gesture occupies at least 80% of the window, accepts rest only when every sample is rest, and discards all other windows. This exactly matches the stated purity rule, so the processed data did not need replacement.

## 3. Temporal voting

`voting.py` is deterministic and NumPy-only. Two interpretations were evaluated:

- `paper_style_vote`: one-window look-ahead; an accepted label changes only when the next prediction supports the change. This removes isolated one-window switches and adds 100 ms host buffering.
- `two_window_debounce_vote`: causal control; a candidate must be observed twice before the state changes.

Results on all 11,276 test windows:

| Protocol | Accuracy | Macro-F1 | Gesture-only accuracy |
|---|---:|---:|---:|
| strict per-window baseline | 83.9305% | 0.6561 | 57.9432% |
| **paper-style look-ahead vote** | **84.3828%** | **0.6693** | **58.9619%** |
| causal two-window debounce | 83.6201% | 0.6611 | 58.2100% |
| supplementary +/-200 ms tolerance | 85.8904% | not reported | not reported |

The look-ahead filter gains 0.4527 percentage points. The causal version loses 0.3104 points, showing that the exact definition of voting matters.

The tolerance metric is intentionally separate. It counts a prediction as correct when it matches any ground-truth label for the same subject within two 100 ms-hop windows. The paper does not publish exact pseudocode for this metric, so 85.8904% is only a supplementary reconstruction.

## 4. Sparsity regularization

### 4.1 Method

Every run warm-started from `delay62_finetune/best.pt`. Topology, bit width, delay range, and decay convention remained unchanged.

```text
loss = spike_rate_loss + lambda_sparsity * mean(firing_rate_across_four_layers)
sparsity = 1 - total_active_spikes / total_possible_spikes
```

Sparsity uses group size 1 and is weighted by the number of neurons in each layer.

### 4.2 One-epoch sensitivity scan

| Lambda | Test accuracy | Sparsity |
|---:|---:|---:|
| 0.0 | 83.97% | 92.12% |
| 0.001 | 83.74% | 93.12% |
| 0.003 | 82.83% | 94.65% |
| 0.01 | 79.46% | 97.19% |
| 0.03 | 71.38% | 98.81% |

The unregularized Delay-SNN is already about 92% sparse because SpikeRate targets are low and hard-reset LIF dynamics suppress excess firing. Strong regularization therefore trades accuracy for relatively little additional sparsity.

### 4.3 Converged Pareto sweep

Each run used at most 10 epochs and `patience=5`. Only `lambda=0.01` stopped early, at epoch 7.

| Lambda | Test accuracy | Macro-F1 | Gesture accuracy | Sparsity | Best epoch |
|---:|---:|---:|---:|---:|---:|
| original checkpoint | 83.9305% | 0.6561 | 57.9432% | 91.93% | n/a |
| 0.0 warm-start control | 83.4516% | 0.6429 | 56.4395% | 92.12% | 10 |
| 0.0005 | 83.3097% | 0.6426 | 56.8276% | 94.03% | 10 |
| **0.001** | **83.4072%** | **0.6433** | **56.7305%** | **95.30%** | 10 |
| 0.002 | 83.1146% | 0.6331 | 55.4693% | 97.19% | 10 |
| 0.005 | 82.5027% | 0.6205 | 53.9898% | 98.21% | 10 |
| 0.01 | 80.9773% | 0.5822 | 50.4730% | 98.70% | 7 |

The zero-regularization warm-start control itself loses 0.4789 points because Adam state was reset and the original checkpoint was already the best point on its training trajectory. Regularization costs should therefore be compared primarily with the 83.4516% control, not attributed entirely relative to 83.9305%.

Against the control, `lambda=0.0005–0.001` changes accuracy by no more than 0.14 points while increasing sparsity to 94.03–95.30%. At `lambda=0.002`, the loss is 0.34 points and sparsity reaches 97.19%. The cost becomes much steeper beyond that value.

### 4.4 Fixed-point evaluation

Deployment-equivalent inference uses INT8 weights, Q8 membrane state, decay `230/256`, round-to-nearest-even, and hard reset.

| Lambda | Floating-point accuracy | RTL-equivalent fixed-point accuracy | Fixed-point gap |
|---:|---:|---:|---:|
| original checkpoint | 83.9305% | 83.0702% | -0.8603 pp |
| 0.001 | 83.4072% | 82.8751% | -0.5321 pp |
| 0.002 | 83.1146% | 81.9085% | -1.2061 pp |

`lambda=0.001` is nearly free in the floating-point comparison, but its deployment-equivalent accuracy is still 0.1951 points below the original fixed-point baseline. Deployment decisions should use this fixed-point result rather than the training-time Pareto table alone.

## 5. Export compatibility

The existing deployment exporter was reused without modification:

```bash
python scripts/export_fixed_point.py \
  --checkpoint <checkpoint-path> \
  --test-data <processed-test.npz> \
  --output <output-directory>/weights \
  --vectors <output-directory>/vectors
```

A clean re-export of the original checkpoint reproduced the deployed manifest accuracy exactly:

```text
fresh RTL-integer evaluation: 0.8307023767293367
existing manifest accuracy:    0.8307023767293367
```

Both regularized candidates exported successfully with the original topology and field names:

```text
lambda=0.001 scales=[0.0267, 0.0402, 0.0502, 0.0220]
             thresholds_q8=[9597, 6375, 5095, 11631]
lambda=0.002 scales=[0.0261, 0.0444, 0.0531, 0.0222]
             thresholds_q8=[9797, 5766, 4825, 11515]
baseline     scales=[0.0233, 0.0365, 0.0483, 0.0173]
             thresholds_q8=[10967, 7021, 5297, 14758]
```

Compatibility is preserved because the model remains `96 -> 64 -> 128 -> 64 -> 13`, delays remain integers in `[0, 62]`, and the exporter fixes decay, rounding, reset, and integer formats independently of the training loss.

## 6. Combined results

### 6.1 Main configurations

| Configuration | Accuracy | Macro-F1 | Gesture accuracy | Sparsity |
|---|---:|---:|---:|---:|
| original FP baseline | 83.9305% | 0.6561 | 57.9432% | 91.93% |
| baseline + paper-style vote | **84.3828%** | 0.6693 | 58.9619% | n/a |
| original RTL-equivalent fixed point | 83.0702% | 0.6317 | 55.9544% | n/a |
| `lambda=0.001`, floating point | 83.4072% | 0.6433 | 56.7305% | 95.30% |
| `lambda=0.001`, fixed point | 82.8751% | 0.6302 | 54.7659% | n/a |
| `lambda=0.002`, floating point | 83.1146% | 0.6331 | 55.4693% | 97.19% |
| `lambda=0.002`, fixed point | 81.9085% | 0.6009 | 52.8014% | n/a |

### 6.2 Regularization plus voting, floating-point path

| Configuration | Before voting | After voting | Change |
|---|---:|---:|---:|
| zero-lambda warm-start control | 83.4516% | 84.2852% | +0.8336 pp |
| `lambda=0.001` | 83.4072% | 84.2054% | +0.7982 pp |
| `lambda=0.002` | 83.1146% | 83.6910% | +0.5764 pp |

Voting improved all three floating-point configurations. The equivalent fixed-point-plus-voting combinations were not measured and must not be inferred as confirmed results.

## 7. Deployment interpretation

- Choose the original checkpoint plus look-ahead voting when accuracy is the priority and 100 ms host buffering is acceptable.
- Consider `lambda=0.001` when a measured sparse hardware scheduler benefits from 95.30% activity sparsity enough to justify a 0.1951-point fixed-point accuracy loss.
- Keep higher lambda values only as Pareto points until real cycle or power savings justify their larger accuracy cost.

This report does not select the final hardware configuration because model sparsity is useful only if the hardware actually skips inactive work.

## 8. Delivered files

```text
semg_snn_fpga_reproduction/
├── voting.py
├── evaluate_voting.py
├── train_sparsity.py
├── summarize_sparsity_sweep.py
├── DELAY_RETRAIN_REPORT.md
└── runs/
    ├── delay62_finetune/voting_metrics.json
    ├── sparsity_lambda_*/best.pt
    ├── sparsity_lambda_*/metrics.json
    ├── sparsity_sweep_summary.json
    ├── sparsity_plus_voting_combo.json
    └── sparsity_sweep.log
```

The public repository may omit private checkpoints, processed data, and generated run artifacts; the paths above document the original experiment layout.

## 9. Claim boundaries

All tabulated accuracy and sparsity values are measured on the stated experiment outputs. The fixed-point values come from the existing deployment-equivalent NumPy path. No new RTL power or timing measurement was performed as part of this retraining study.
