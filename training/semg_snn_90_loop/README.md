# NinaPro DB5 13-Class SNN Optimization

This project develops the original 83.93% delay-encoded SNN baseline under a
fixed NinaPro DB5 Exercise A protocol. The final results are:

- **91.10%** for a fully online stream without gesture boundaries or future
  information;
- **92.03%** for continuous windows when repetition boundaries are known;
- **93.32%** for a filtered-segment protocol that resets state at gaps.

The 91.10% result is the most credible number for real-time deployment. The
other two results are useful for comparison with common offline preprocessing
protocols but are not equivalent online results. See [`RESULTS.md`](RESULTS.md)
for definitions and failed experiments and [`RESEARCH.md`](RESEARCH.md) for the
literature review.

## Data and split

- Dataset: NinaPro DB5, Exercise A.
- Classes: Rest plus 12 finger gestures (13 classes).
- Sampling: 200 Hz, 16 channels.
- Window: 100 samples (500 ms), shift 20 samples (100 ms), 80% overlap.
- Training: repetitions 1/2/4/6.
- Validation: repetition 3.
- Test: repetition 5.
- Samples: 44,630 / 11,060 / 11,276.
- Checkpoints, fusion weights, and the Rest bias are selected on validation
  data only.

Rest accounts for 63.44% of the test set, so overall accuracy must be reported
with macro-F1 and gesture-only accuracy.

## Environment

The experiments used Python 3.11, PyTorch 2.5.1, CUDA 12.4, an NVIDIA RTX 3080,
NumPy 2.2.6, scikit-learn 1.8.0, and SciPy 1.16.3. Use the repository-level
`requirements.txt`; no workstation-specific environment path is required.

## Reproduce the strict online route

Run from this directory:

```bash
# Generated data can be reused. Run these only when rebuilding it.
python prepare.py
python prepare_continuous_context.py

# A 23-window (2.2 s additional history) PLIF context model.
python train.py \
  --model context_class_adaptive \
  --context 23 \
  --continuous-context \
  --stream-context \
  --epochs 24 \
  --patience 7 \
  --batch-size 256 \
  --lr 0.0002 \
  --run-name context23_class_plif_stream \
  --init runs/context23_class_plif_continuous/best.pt

# Fuse the Context model with a ConvLIF/Jaccard SNN and the Delay-SNN.
python evaluate_ensemble.py \
  context_class_adaptive_stream,23,runs/context23_class_plif_stream/best.pt \
  hybrid,1,runs/hybrid_sja_v1/best.pt \
  delta,3,../semg_snn_fpga_reproduction/runs/delay62_finetune/best.pt \
  --output runs/strict_stream_three_expert_metrics.json

# Audit causal smoothing over the complete continuous stream.
python evaluate_full_stream.py \
  --context 23 \
  --checkpoint runs/context23_class_plif_stream/best.pt \
  --output runs/strict_full_stream_context23_metrics.json \
  --probabilities runs/strict_full_stream_context23_probabilities.npz
```

The validation set selects smoothing width 1, meaning that additional temporal
smoothing hurts because it delays gesture transitions.

## Key files

```text
prepare.py                         Strict split, raw windows, and 336 features
prepare_continuous_context.py      Features and indices for every continuous window
model.py                           Feature, Context, PLIF, ConvLIF, Jaccard, and hybrid SNNs
train.py                           Training, early stopping, streaming/boundary modes
evaluate_ensemble.py               Validation-selected weights, Rest bias, and test metrics
evaluate_full_stream.py            Boundary-free full-stream causal audit
hw_ops.py / hw_model.py            Hardware-friendly operators and QAT models
train_qat.py                       Hardware-friendly quantization-aware training
export_hw_fixed.py                 Fixed-point model export
hw_fixed_reference.py              NumPy reference inference without PyTorch
runs/*/best.pt                     Best checkpoints (not included in Git)
runs/*metrics.json                 Machine-readable experiment results
```

## Recommended interpretation

For offline paper comparisons, 93.32% may be reported only with the explicit
statement that filtered segments and gap resets provide boundary information.
For a real-time wearable or prosthetic controller, use 91.10% as the primary
result and validate it again on truly continuous acquisition, unknown gesture
transitions, different days, and unseen users.
