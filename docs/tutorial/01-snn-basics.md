# 01. LIF Neurons and Surrogate Gradients

## 1.1 ANN neuron versus SNN neuron

An ordinary artificial neuron produces a continuous activation in one pass:

```text
y = activation(Wx + b)
```

A spiking neuron carries state across time. A discrete leaky integrate-and-fire neuron can be written as:

```text
mem[t] = beta * mem[t-1] + input[t]
spike[t] = 1 if mem[t] >= threshold else 0
mem[t] = mem[t] - spike[t] * threshold
```

`beta` controls how long the neuron remembers earlier input. The output is binary, but information can be represented by spike count, timing, or both.

## 1.2 Minimal simulation

```python
def lif_step(current, membrane, beta=0.9, threshold=1.0):
    membrane = beta * membrane + current
    spike = float(membrane >= threshold)
    membrane = membrane - spike * threshold  # soft reset
    return spike, membrane

membrane = 0.0
for t in range(20):
    spike, membrane = lif_step(0.3, membrane)
    print(t, spike, round(membrane, 4))
```

With a constant input, the membrane accumulates charge, emits a spike after crossing the threshold, and retains the excess charge after a soft reset.

## 1.3 Why ordinary backpropagation fails

The threshold function has zero derivative almost everywhere and is undefined at the threshold. Using its true derivative therefore blocks useful gradients.

A surrogate gradient keeps the hard threshold in the forward pass but substitutes a convenient derivative in the backward pass. The simplest straight-through estimator (STE) passes the gradient unchanged:

```python
import torch

class SpikeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

spike = SpikeSTE.apply
```

Smoother triangular, sigmoid, or arctangent surrogate derivatives are also common. The surrogate is a training device; inference still uses an exact binary comparison.

## 1.4 Summary

- LIF neurons integrate input through time, leak old state, spike, and reset.
- The threshold makes native gradients unusable.
- Surrogate gradients make SNNs trainable while preserving hard spikes at inference.

Next: [02-first-training.md](02-first-training.md)
