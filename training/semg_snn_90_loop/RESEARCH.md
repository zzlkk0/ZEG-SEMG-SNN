# SNN + sEMG Literature and Route Mapping

Search date: 2026-07-28. Paper pages, author manuscripts, and official code are
preferred below. Reported values are directly comparable only when the dataset,
class count, split, and window protocol match.

## Directly relevant work

### SpGesture: Spiking Jaccard Attention

- Paper: [NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/409334f42cbb57d07aa152f2d0433ec7-Abstract-Conference.html)
- Preprint: [arXiv:2405.14398](https://arxiv.org/abs/2405.14398)
- Code: [guoweiyu/SpGesture](https://github.com/guoweiyu/SpGesture)

The main components are ConvLIF encoding/feature extraction, channel-level
Spiking Jaccard Attention, and an LIF classifier. The paper reports 89.26% on a
self-collected posture-shift dataset, not NinaPro DB5. This project adapts the
ConvLIF and intersection-over-union attention ideas. The single-window model
improves from 87.41% for Feature-SNN to 88.76% and contributes more strongly
when fused with Context models.

### Low-power FPGA sEMG SNN

- 2024 work: [sEMG-based gesture recognition with SNNs on low-power FPGA](https://iris.unica.it/handle/11584/469172)
- Follow-up manuscript: [Real-Time sEMG Processing with SNNs on a Low-Power 5K-LUT FPGA](https://iris.unica.it/retrieve/33ced2bf-a130-4a5e-8bdd-671978405242/Real-Time_sEMG_Processing_with_Spiking_Neural_Networks_on_a_Low-Power_5K-LUT_FPGA.pdf)
- Official training repository: [EOLAB SNN-sEMG-GestureClassification-ForceTracking](https://github.com/eolabcolab/SNN-sEMG-GestureClassification-ForceTracking)

The 2024 version reports 85.6% on DB5. The follow-up reports 83.17% for
12 gestures and includes FPGA power/energy measurements. Its main value is
hardware feasibility rather than maximum accuracy. This project retains a
small, approximately 23.6k-parameter Delay-SNN as a low-power expert.

### Learnable membrane time constants (PLIF)

- Paper: [Incorporating Learnable Membrane Time Constant to Enhance Learning of SNNs](https://arxiv.org/abs/2007.05785)

PLIF shows that learnable membrane time constants improve temporal expression
in directly trained SNNs. This project adds layer-level learnable decay,
neuron-level decay offsets, and a separate history decay for every class.
Heterogeneous class/neuron time constants improve the known-boundary continuous
single model from 89.03% to 89.42% and improve ensemble gesture accuracy.

### Temporal Efficient Training (TET)

- Paper: [ICLR 2022 / arXiv:2202.11946](https://arxiv.org/abs/2202.11946)
- Official code: [brain-intelligence-lab/temporal_efficient_training](https://github.com/brain-intelligence-lab/temporal_efficient_training)

TET optimizes loss more consistently over all time steps and can address weak
supervision of early states. It was not fully ported because Context-SNN uses a
window-weighted readout, but it remains a high-value future training objective.

### Adaptive multi-delta and TAD-LIF

- Paper: [High-speed Low-consumption sEMG-based Transient-state micro-Gesture Recognition](https://arxiv.org/abs/2403.06998)

The paper proposes adaptive multi-delta encoding and TAD-LIF and reports 83.85%
and 93.52% on two self-collected datasets. An initial adaptive-delta experiment
in this project reached about 80.5% DB5 validation accuracy and was stopped. It
is not an exact reproduction of the paper.

### Spiking Transformer / ESTU

- Manuscript: [ESTU: Enabling Spiking Transformers on Ultra-Low-Power FPGAs](https://iris.unica.it/retrieve/e43af17f-5826-4407-850d-ff4a9f89bcff/ESTU_Enabling_Spiking_Transformers_on_Ultra-Low-Power_FPGAs%20%281%29.pdf)
- Code: [EOLAB-2025/ESTU](https://github.com/EOLAB-2025/ESTU)

ESTU reports approximately 87.21% on DB5. It is particularly relevant for
joint accuracy-power-resource optimization. A direct implementation of this
project's 91% ensemble has substantially more computation than the small FPGA
Delay-SNN.

### Spiking reservoir

- Record: [Event-driven physical reservoir computing for sEMG](https://eprints.gla.ac.uk/360949/)
- DOI: [10.1109/TAI.2025.3592899](https://doi.org/10.1109/TAI.2025.3592899)

The all-spiking rotating-neuron reservoir reports roughly 80.3%. Its advantage
is near-sensor, low-latency, hardware-friendly inference rather than peak DB5
accuracy. It is a useful baseline when ultra-low power matters more than the
complex Context-SNN.

## Non-SNN upper bounds and transferable ideas

These studies indicate whether 90% is plausible, but are not direct fair
comparisons with this protocol.

- [sEMGXCM](https://www.mdpi.com/2306-5354/10/9/1101) reports approximately
  92.3%/94.2% in a subject-specific 53-class DB5 setting, so 90% is not a hard
  dataset ceiling.
- [STCNet](https://doi.org/10.1016/j.compbiomed.2024.109525) combines
  spatiotemporal crossing with subject-aware contrastive learning. Official
  code: [KNU-BrainAI/STCNet](https://github.com/KNU-BrainAI/STCNet).
- [A simple sEMG attention model](https://arxiv.org/abs/2006.03645) shows that
  conventional preprocessing and compact attention can form strong baselines.
- A [2026 few-shot prototype-adaptation study](https://www.nature.com/articles/s41598-026-40352-6)
  reports about 97.6% in a subject-specific DB5 setting. It supports user
  calibration, but the protocol differs. Simple per-subject fusion calibration
  in this project did not outperform global fusion.

## Literature-to-implementation route

```text
Original Delay/Delta SNN (83.93%)
        |
        +-- Handcrafted time/frequency features + Feature-SNN
        +-- ConvLIF + Spiking Jaccard Attention
        +-- Causal Context-SNN
        |      +-- PLIF and class-heterogeneous time constants
        +-- Validation-selected probability fusion + one Rest bias
               +-- Boundary-free online: 91.10%
               +-- Known boundaries:      92.03%
               +-- Filtered segments:     93.32%
```

## Priorities for future work

1. **Freeze one protocol:** use `--stream-context` as the sole online protocol;
   prohibit repetition boundaries and filtered-window smoothing.
2. **TET or per-time-step auxiliary loss:** directly supervise early PLIF
   transition states to reduce contamination from the 2.2 s history.
3. **Joint gesture detection and classification:** detect Rest/transition first,
   then classify 12 gestures, using past information only.
4. **Contrastive pretraining:** adapt subject-aware supervised contrastive
   learning from STCNet, then distill the representation into an SNN.
5. **Unseen-subject and cross-day tests:** add LOSO, cross-session, and electrode
   displacement tests.
6. **ANN-to-SNN distillation:** use a strong sEMGXCM/STCNet-like teacher while
   retaining an SNN at inference.
7. **Quantization and energy:** unify input encodings and report fixed-point
   accuracy, spike sparsity, MAC/AC counts, latency, and FPGA resources. Current
   accuracy alone does not establish a low-power advantage.
