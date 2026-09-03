# 04. Parallel Branches and Fusion

## 4.1 Why not use one larger model?

A wider model still sees the same representation and often preserves the same failure modes. Complementary branches can observe global features, local waveform patterns, and delayed temporal structure. Their errors are less correlated, so validation-selected fusion can outperform every individual branch.

## 4.2 Two complementary views

For a compact demonstration, reshape 20 features into a pseudo sequence with 10 time points and 2 channels:

```python
def to_pseudo_sequence(x, time=10, channels=2):
    return x.view(x.shape[0], time, channels)
```

A Context branch can process the complete feature vector with a fully connected LIF layer. A Conv branch can process the pseudo sequence with `Conv1d` followed by LIF dynamics. Train them independently, with separate optimizers and validation-selected checkpoints.

In the full project, the three views are Context PLIF, Hybrid ConvLIF/Jaccard, and Delay-SNN.

## 4.3 Probability fusion

Convert each branch's logits to probabilities and combine them with non-negative weights that sum to one:

```python
import numpy as np
import torch.nn.functional as F

def search_two_branch_weight(a_val, b_val, y_val, resolution=20):
    pa = F.softmax(a_val, dim=1).cpu().numpy()
    pb = F.softmax(b_val, dim=1).cpu().numpy()
    y = y_val.cpu().numpy()
    best = (-1.0, None)
    for weight_a in np.linspace(0.0, 1.0, resolution + 1):
        prediction = (weight_a * pa + (1.0 - weight_a) * pb).argmax(1)
        accuracy = (prediction == y).mean()
        if accuracy > best[0]:
            best = (accuracy, weight_a)
    return best
```

For three branches, search a two-dimensional simplex. Perform the search on validation data only, freeze the selected weights, and then evaluate the test set once.

## 4.4 Rest-class calibration

When the rest class dominates the data, a small validation-selected bias can correct systematic overprediction:

```python
def calibrate_rest_bias(probability, labels, biases, rest_class=0):
    best = (-1.0, 0.0)
    for bias in biases:
        adjusted = probability.copy()
        adjusted[:, rest_class] *= np.exp(bias)
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        accuracy = (adjusted.argmax(1) == labels).mean()
        if accuracy > best[0]:
            best = (accuracy, bias)
    return best
```

The calibration parameter is part of model selection, so it must also be chosen without test-set feedback.

## 4.5 Project result

The full sEMG system's individual branches achieved roughly 84–89% strict accuracy. Three-branch fusion plus a validation-selected rest bias exceeded 91%, demonstrating the value of complementary error patterns. These figures are project results, not targets guaranteed by the simplified tutorial.

## 4.6 Summary

- Diversity matters more than merely enlarging one branch.
- Train branches independently and fuse calibrated probabilities.
- Select weights and biases only on validation data.
- Host-side fusion can keep softmax and calibration outside the FPGA.

Next: [05-quantization-motivation.md](05-quantization-motivation.md)
