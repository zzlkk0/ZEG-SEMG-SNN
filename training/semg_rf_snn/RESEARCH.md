# SNN-related routes toward 90% on sEMG

## Most relevant result

Manna, Bihl, and Di Caterina, “Resonate-and-fire neurons meet EMG”
(ICASSP 2026) reports **92.36%** on NinaPro DB5 Exercise A (13 classes).
Its key idea is an RF bank spanning 20--100 Hz, followed by a 3-layer LIF
SNN. The reported ablations are:

| Setting | Accuracy |
|---|---:|
| 8 channels, 80 RF neurons, delay | 88.05% |
| 16 channels, 80 RF neurons, delay | 92.33% |
| 16 channels, 80 RF neurons, no delay | 92.20% |
| 16 channels, 20 RF neurons, delay | 88.69% |
| 16 channels, 160 RF neurons, delay | 92.36% |

Important caveat: repetitions 2 and 5 are both used as validation, while
1, 3, 4, and 6 are training. It does not report a separate held-out test set.
The paper also uses 52-sample windows with 80% overlap and random temporal
shifts of -8 to +8 samples.

Primary source:
https://strathprints.strath.ac.uk/95370/

## Other relevant work

- SpGesture (NeurIPS 2024) introduces Jaccard attentive SNNs and source-free
  domain adaptation. It reports 89.26% on a separate posture-shift dataset,
  so the number is not directly comparable to DB5. Its public repository
  currently contains a placeholder README rather than released code.
  https://papers.nips.cc/paper_files/paper/2024/hash/409334f42cbb57d07aa152f2d0433ec7-Abstract-Conference.html
  https://github.com/guoweiyu/SpGesture/

- Donati et al., “Low Power Neuromorphic EMG Gesture Classification,” reports
  about 90% using a hybrid recurrent SNN with LIF/DEXAT neurons and
  multi-threshold onset/offset encoding. It is a **3-class Roshambo** task,
  not the present 13-class DB5 task.
  https://arxiv.org/abs/2206.02061

- Scrugli et al.'s FPGA-oriented DB5 pipeline and its later real-time FPGA
  version remain in the low/mid-80% range. This is consistent with the
  83.93% strict-test result in the existing reproduction.
  https://iris.unica.it/retrieve/5ed2d22a-6981-4379-99d7-0e6f8089b772/Real-Time_sEMG_Processing_With_Spiking_Neural_Networks_on_a_Low-Power_5K-LUT_FPGA.pdf

## Practical conclusion

90% is plausible for an SNN-related DB5 Exercise-A system, but the only
direct 90%+ evidence found uses RF frequency encoding and reports validation
rather than a separate held-out test. Under the stricter repetition-5 test,
the present first RF-SNN implementation did not reproduce the gain.

The next faithful experiment should recreate the paper's 52-sample window
generation directly (rather than center-cropping audited 100-sample windows),
apply the stated ±8-sample augmentation, use graded RF events with the paper's
Bayesian-optimized threshold/scale, and compare both its split and the strict
split side-by-side.
