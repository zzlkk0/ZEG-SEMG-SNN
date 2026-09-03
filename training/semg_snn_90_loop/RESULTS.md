# Two-Hour SNN Optimization Results

## Outcome

Starting from the strict 83.9305% Delay-SNN baseline, the project reached three
different protocol-dependent results:

| Protocol | Reads gesture boundaries | Uses future information | Test accuracy | Macro-F1 | Gesture accuracy |
|---|---:|---:|---:|---:|---:|
| Fully boundary-free online stream | No | No | **91.0961%** | 0.8235 | 79.7720% |
| Continuous windows with known boundaries | Yes | No | **92.0273%** | 0.8415 | 81.9064% |
| Filtered segments with gap resets | Yes | No | **93.3221%** | 0.8597 | 84.5743% |

The target of exceeding 90% was therefore achieved even under the strictest,
most deployment-relevant online protocol. This is still a cross-repetition
evaluation on the same subjects, not leave-one-subject-out generalization.

## Shared evaluation conditions

- 13 classes: Rest plus 12 Exercise-A finger gestures.
- Train repetitions: 1/2/4/6; validation: 3; test: 5.
- 500 ms windows, 100 ms hop, 80% overlap.
- Normalization uses training statistics only.
- Checkpoints, fusion weights, and one Rest logit bias are selected on
  validation data.
- The test split contains 11,276 windows; 7,153 (63.44%) are Rest.

Although repetition 5 is excluded from optimization, the test results were
observed repeatedly while deciding what to try next. The 91.10% figure should
therefore be treated as a development-stage final result. A formal paper should
lock the method and add untouched cross-day or cross-subject evaluation.

## Route A: fully boundary-free online stream

This is the recommended primary result. Context indices can access only earlier
windows from the same subject. They do not inspect repetition identifiers and
do not reset at gesture transitions.

The three SNN experts are:

1. a class-adaptive PLIF Context-SNN using 23 windows (2.2 s extra history);
2. a single-window ConvLIF + Spiking Jaccard Attention SNN;
3. a single-window delay/Delta-encoded SNN.

Validation selected probability weights **0.5 / 0.4 / 0.1**:

| Metric | Validation | Test |
|---|---:|---:|
| Uncalibrated accuracy | 91.5461% | 90.7414% |
| Uncalibrated macro-F1 | 0.8508 | 0.8195 |
| Accuracy after validation-selected Rest bias = -0.55 | 91.8897% | **91.0961%** |
| Calibrated macro-F1 | 0.8545 | **0.8235** |
| Calibrated gesture accuracy | 82.8273% | **79.7720%** |

The full result is stored as
`runs/strict_stream_three_expert_metrics.json` when reproduced locally. A
two-expert ensemble without the old Delta branch reaches 90.8478%, showing that
the gain does not come entirely from the baseline model.

### Full continuous-stream smoothing audit

`evaluate_full_stream.py` processes 72,708 continuous windows and resets only
at subject boundaries. Validation compared widths 1/3/5/7/9/11/15/21/31 and
selected **width 1**. Any additional smoothing reduced accuracy because of
transition lag. High scores from filtered windows must not be presented as
online smoothing results.

## Route B: continuous windows with known segment boundaries

This protocol retains transition windows but allows Context construction to
read repetition boundaries and block cross-segment history. It applies when an
external detector supplies transition signals.

The four experts use weights 0.25 / 0.20 / 0.35 / 0.20:

- a 31-window class-adaptive PLIF model;
- a fixed 15-window Context-SNN;
- a single-window ConvLIF/Jaccard model;
- the Delta-SNN.

Test accuracy is 91.7435% before calibration and **92.0273%** after applying the
validation-selected Rest bias of -0.45. The detailed artifact is
`runs/known_boundary_continuous_best_metrics.json`.

## Route C: filtered segments with gap resets

This protocol constructs history only over windows retained by label filtering
and resets whenever the timeline has a gap. It resembles many offline window
classification papers but provides additional segment-boundary information.

Validation selected weights **0 / 0.5 / 0.4 / 0.1** for Context-15,
Context-23, ConvLIF/Jaccard, and Delta respectively. Test accuracy is 92.9762%
before calibration and **93.3221%** after a Rest bias of -0.475. The detailed
artifact is `runs/filtered_segment_four_expert_metrics.json`.

## Single-model progression

The first rows use the filtered-segment/gap-reset protocol unless marked
otherwise.

| Model | Protocol | Accuracy | Macro-F1 | Gesture accuracy |
|---|---|---:|---:|---:|
| Original Delay-SNN | Original strict split | 83.9305% | 0.6561 | 57.94% |
| Feature-SNN | Single window | 87.4069% | 0.7287 | 69.415% |
| ConvLIF + Jaccard Hybrid-SNN | Single window | 88.7637% | 0.7615 | 72.423% |
| Context-5 | Filtered segment | 89.2870% | 0.7705 | 74.024% |
| Context-7 | Filtered segment | **90.0585%** | 0.7879 | 75.891% |
| Context-9 | Filtered segment | 90.3335% | 0.7965 | 76.255% |
| Context-15 | Filtered segment | 91.1671% | 0.8132 | 78.753% |
| Context-23 | Filtered segment | 91.2912% | 0.8141 | 78.729% |
| Adaptive Context-23 | Filtered segment | **91.7701%** | 0.8259 | 80.063% |
| Class-PLIF Context-23 | Continuous, known boundaries | 89.4200% | 0.7985 | 77.516% |
| Context + ConvLIF residual | Continuous, known boundaries | 89.4643% | 0.7979 | 77.880% |
| Class-PLIF Context-23 | Fully boundary-free | **88.5332%** | 0.7822 | 73.757% |

A Context-7 model exceeds 90% under the filtered-segment protocol. Under the
fully boundary-free protocol, multiple complementary experts are needed to
exceed 90% reliably.

## Changes that helped

- 336 time- and frequency-domain features with training-subject normalization.
- ConvLIF raw-waveform encoding and Spiking Jaccard Attention.
- Causal Context-SNN and learnable PLIF decay.
- Class-specific history decay and heterogeneous neuron time constants.
- Validation-selected probability fusion across time scales and encodings.
- A single Rest logit bias to address the 63.44% Rest imbalance.

## Routes that did not provide stable gains

| Route | Outcome |
|---|---|
| Cepstral feature concatenation | Test 85.41%; clear regression |
| Initial adaptive multi-delta | Validation about 80.5%; stopped early |
| Mixup Context-SNN | Lower validation performance |
| Strong time-shift/noise/channel dropout | Hybrid test about 88.61%; no stable gain |
| Mean+max ConvLIF pooling | Test about 88.52%, below mean pooling |
| Spatial ConvLIF branch | Spike activity collapsed |
| Subject-FiLM | Approximately tied with the global model |
| Per-subject models | Aggregated test 91.68%, below global filtered ensemble |
| Per-subject fusion calibration | 91.90% under known boundaries, below 92.03% global |
| 65-parameter classwise fusion | 91.54% after subject-grouped CV |
| 1.4 s boundary-free fixed Context | Test 85.31%, far below PLIF |
| Full-stream causal smoothing | Validation chose width 1; smoothing should not be used |

## Limitations

1. Training, validation, and test contain the same ten subjects.
2. Window-level metrics with 80% overlap do not equal independent-trial success.
3. The original results are PyTorch FP32 rather than FPGA measurements.
4. The 93.32% route uses filtered-segment boundaries; the 92.03% route uses
   known repetition boundaries.
5. The 91.10% route is boundary-free but still requires validation on real
   continuous acquisition, cross-day drift, and false triggers.
6. The old Delta preprocessing and the new feature-normalization path have
   different state semantics and should be unified before deployment.
