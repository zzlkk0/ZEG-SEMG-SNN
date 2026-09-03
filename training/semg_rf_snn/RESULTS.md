# Results

## Strict held-out repetition split

- Classes: 13
- Train: repetitions 1, 2, 4, 6 — 44,630 windows
- Validation: repetition 3 — 11,060 windows
- Test: repetition 5 — 11,276 windows
- RF window: center 52 samples (260 ms) of each audited 100-sample window
- Best epoch: 14
- Validation accuracy: **81.19%**
- Validation macro-F1: **61.84%**
- Test accuracy: **80.00%**
- Test macro-F1: **56.54%**

This first RF-SNN approximation did not beat the delayed delta-encoded SNN
baseline (83.93%). It overfit after epoch 14. Consequently, the ICASSP 2026
paper's 92.36% result cannot yet be claimed under our strict test protocol.

Checkpoint: `runs/strict_split/best.pt`

## Faithful 52-sample temporal RF-SNN

This follow-up uses windows generated directly from the continuous recordings:
52 samples, stride 10 (~80.8% overlap), random training shifts in [-8, 8],
80 RF frequencies from 20--100 Hz, graded phase-crossing events, and a
two-hidden-layer temporal LIF classifier.

| Protocol | Best epoch | Validation accuracy | Validation macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|---:|
| Paper-like: train 1,3,4,6; validate 2,5 | 9 | **78.25%** | 59.13% | n/a | n/a |
| Strict: train 1,2,4,6; validate 3; test 5 | 5 | **79.83%** | 61.72% | **80.24%** | 58.91% |

The RF input activity is only about 1.5%, but neither protocol reproduces the
paper's 92.36%. Training loss continued to decrease while validation accuracy
fell, indicating cross-repetition overfitting rather than under-training.

Checkpoints:

- `runs/faithful_paper_rf80/best.pt`
- `runs/faithful_strict_rf80/best.pt`

The implementation follows all reported high-level settings, but is not a
bit-exact Lava/SLAYER reproduction: the paper does not publish source code,
the exact Bayesian-optimized mapping of threshold/decay to RF neurons, the
class-weight scale gamma, or the two hidden-layer widths.
