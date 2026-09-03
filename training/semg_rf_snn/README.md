# RF-SNN for NinaPro DB5 Exercise A

This project tests the frequency-sensitive Resonate-and-Fire (RF) idea from
Manna et al. (ICASSP 2026) while retaining the strict split used by the
existing FPGA-SNN reproduction:

- train: repetitions 1, 2, 4, 6
- validation: repetition 3
- test: repetition 5
- classes: rest + 12 Exercise-A gestures (13 classes)

The RF frontend is fixed and differentiable: a bank of damped complex
resonators spanning 20--100 Hz converts each 260 ms raw-EMG segment into
graded frequency spikes. A two-hidden-layer LIF SNN performs classification.

The source paper reports 92.36% validation accuracy, but uses repetitions 2
and 5 for validation and the remaining repetitions for training. Results here
therefore should not be treated as an exact reproduction.

## Run

```bash
python prepare_raw.py
python train.py --amp

# Direct 52-sample temporal reproduction
python prepare_faithful.py --protocol paper
python prepare_faithful.py --protocol strict
python train_faithful.py --protocol paper
python train_faithful.py --protocol strict
```
