# From SNN Training to FPGA Deployment: Tutorial Series

This course builds a small spiking neural network from first principles, then develops it into a multi-branch sEMG classifier and prepares the inference path for FPGA deployment. The examples are intentionally compact, while the accompanying notebooks connect each idea to the full NinaPro DB5 project.

## What you will learn

- how a leaky integrate-and-fire (LIF) neuron evolves over time
- why spikes need a surrogate gradient during training
- how to train single- and multi-layer SNN classifiers
- how validation-only fusion improves complementary branches
- why floating-point operators are expensive on an FPGA
- how to replace operators and apply quantization-aware training (QAT)
- how to export integer weights and verify them with NumPy
- how to estimate resources without presenting estimates as implementation results

## Chapters

| File | Topic | Main implementation |
|---|---|---|
| [01-snn-basics.md](01-snn-basics.md) | LIF neurons and surrogate gradients | membrane simulation for one neuron |
| [02-first-training.md](02-first-training.md) | First SNN classifier | one-layer LIF with rate encoding |
| [03-deeper-multistep.md](03-deeper-multistep.md) | Deeper multi-step SNNs | two LIF layers, early stopping, gradient clipping |
| [04-parallel-branches-fusion.md](04-parallel-branches-fusion.md) | Parallel branches and fusion | two complementary views and validation-set weight search |
| [05-quantization-motivation.md](05-quantization-motivation.md) | Why quantization is necessary | concepts and memory budgeting |
| [06-hw-friendly-ops-qat.md](06-hw-friendly-ops-qat.md) | Hardware-friendly operators and QAT | fake quantization, operator replacement, QAT loop |
| [07-export-numpy-verify.md](07-export-numpy-verify.md) | Export and NumPy verification | packed weights and independent fixed-point inference |
| [08-resource-estimate-deployment.md](08-resource-estimate-deployment.md) | Resource estimates and deployment | BRAM calculation and implementation checklist |

## Suggested learning path

```text
LIF dynamics -> first classifier -> deeper SNN -> branch fusion
             -> quantization -> QAT -> integer export -> FPGA planning
```

Read chapters 01–04 if your priority is model training. Read 05–08 if your priority is deployment. For reproducible research, follow the full sequence because deployment constraints should influence training decisions.

## Prerequisites

Basic Python, NumPy, and PyTorch are enough. Familiarity with neural-network training helps, but no prior FPGA experience is required.

## Environment

```bash
conda activate torch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Project notebooks

| Notebook | Related chapters | Purpose |
|---|---|---|
| [01_context_training.ipynb](notebooks/01_context_training.ipynb) | 01–03 | train the Context PLIF branch with two-stage curriculum learning |
| [02_context_quantization.ipynb](notebooks/02_context_quantization.ipynb) | 05–07 | Context HW-QAT and NumPy cross-check |
| [03_hybrid_training.ipynb](notebooks/03_hybrid_training.ipynb) | 01–04 | train the ConvLIF + Spiking Jaccard Hybrid branch |
| [04_hybrid_quantization.ipynb](notebooks/04_hybrid_quantization.ipynb) | 06–07 | Hybrid HW-QAT, BatchNorm folding, and reciprocal lookup |
| [05_delay_training.ipynb](notebooks/05_delay_training.ipynb) | 01–03 | train the Delay-SNN baseline and fine-tune learnable axonal delays |
| [06_delay_quantization.ipynb](notebooks/06_delay_quantization.ipynb) | 05, 08 | post-training quantization of Delay-SNN and comparison with board figures |
| [07_fusion_fp32.ipynb](notebooks/07_fusion_fp32.ipynb) | 04 | reproduce 91.10% FP32 three-branch fusion |
| [08_fusion_hw_qat.ipynb](notebooks/08_fusion_hw_qat.ipynb) | 04, 08 | reproduce 91.11% quantized fusion and estimate resources |

The notebooks expect the private dataset and checkpoints described by the project documentation; those artifacts are not included in this public repository.

## Methodological rule

Use the training set to optimize parameters, the validation set to select checkpoints and fusion settings, and the test set once for the final report. Keep measured board or synthesis results separate from analytical estimates.
