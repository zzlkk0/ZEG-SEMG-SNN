# Specification: Retraining the Delay-SNN Toward the Paper Result

## 1. Scope and objective

This task concerns only the Delay-SNN branch. The goal is to move its evaluation closer to the result reported by Scrugli et al. at DASIP 2024 while preserving the deployed topology, fixed-point format, and operation count.

- Fixed topology: `96 -> 64 -> 128 -> 64 -> 13`
- Fixed operation estimate: 2,336,000 MOPS-accounting operations per inference under the existing convention
- Paper result, including temporal voting: 85.6%
- Current strict independent-test result: 83.9305% on 11,276 windows without smoothing

The work has two parts: align training/evaluation details that can be reconstructed from the paper, and add an explicit spike-sparsity regularizer. Do not change the topology or deployment format.

## 2. Known paper settings

The paper describes SLAYER/Lava training with:

- Adam, learning rate 0.001
- batch size 32
- early stopping after 10 epochs without improvement
- SpikeRate targets of 0.2 for the true class and 0.03 for other classes
- gesture windows accepted only when at least 80% of samples carry that gesture label
- rest windows accepted only when 100% of samples are rest
- ambiguous windows discarded

Temporal voting suppresses isolated label changes. The exact frame-level pseudocode is not public, so every reconstructed variant must be named and described precisely.

## 3. Required experiments

### 3.1 Evaluation protocols

Always report the strict per-window result. Also implement a deterministic NumPy paper-style filter that changes the accepted class only after two consecutive windows support the new class. State whether the algorithm is causal or uses one-window look-ahead.

An optional +/-200 ms tolerance metric may be included, but it must be clearly labeled as supplementary and must never replace strict accuracy.

### 3.2 Training-recipe audit

Compare the existing run against the settings above. Record optimizer, loss, learning rate, batch size, epoch count, early stopping, augmentation, and seed. Retrain only if an actual mismatch is found.

### 3.3 Window-label audit

Verify the 80% gesture-purity and 100% rest-purity rules in the preprocessing source and processed data. If the existing data differ, prepare a parallel dataset and report both results rather than overwriting the original split.

### 3.4 Spike-sparsity regularization

Add an L1-style penalty on the mean firing rate across all four layers:

```text
loss = spike_rate_classification_loss + lambda_sparsity * mean_layer_firing_rate
```

Report sparsity using group size 1:

```text
sparsity = (1 - active_spikes / possible_spikes) * 100%
```

Measure the baseline and every regularized checkpoint. If a candidate loses more than 1–2 percentage points, keep it in the Pareto table but do not present it as the recommended deployment model.

### 3.5 Hard compatibility constraints

- topology remains `96 -> 64 -> 128 -> 64 -> 13`
- the first three layers have integer delays in `[0, 62]`; the output layer has no trainable delay
- symmetric signed INT8 weights
- signed Q8 membrane representation
- decay is `230/256`, or another explicitly reported `k/256`
- signed round-to-nearest-even
- hard reset
- thresholds fit the existing RTL fixed-point representation

### 3.6 Full test matrix

Evaluate all 11,276 independent test windows and report accuracy, macro-F1, gesture-only accuracy, and sparsity for:

1. baseline strict
2. baseline plus aligned training, if retraining was needed
3. baseline plus temporal voting
4. sparsity-regularized candidates
5. useful combinations of regularization and voting
6. fixed-point equivalents of deployment candidates

## 4. Acceptance criteria

- The strict baseline is reproduced from a clean run.
- Evaluation filtering is deterministic and NumPy-only.
- Training and labeling rules are audited against source code, not assumed.
- At least several regularization strengths are compared as a Pareto sweep.
- Deployment candidates export through the existing fixed-point tool without topology or RTL changes.
- All negative or neutral results are reported alongside improvements.

## 5. Deliverables

- checkpoints and training metadata for every completed sweep point
- NumPy temporal-voting implementation
- strict, voted, tolerance, sparse, and fixed-point metrics
- sparsity/accuracy Pareto table
- compatibility check for weight scales, thresholds, delays, decay, rounding, and reset
- a concise report that distinguishes measured results from expectations

## 6. Out of scope

This task does not include Context/Hybrid retraining, RTL power measurement, duty-cycle hardware scheduling, or a change of FPGA device.
