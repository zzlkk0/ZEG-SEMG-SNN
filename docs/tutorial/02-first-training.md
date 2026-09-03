# 02. Training Your First SNN Classifier

This chapter trains a one-layer LIF classifier on a synthetic dataset. It is small enough to inspect but contains the same core steps used by larger models.

## 2.1 Synthetic data

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_classification(
    n_samples=6000, n_features=20, n_informative=14,
    n_redundant=2, n_classes=5, random_state=7,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=7,
)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

Fit preprocessing on the training set only. Reusing test statistics leaks information.

## 2.2 Rate encoding

Map each feature to `[0, 1]`, then sample one Bernoulli spike per simulation step:

```python
import torch

def rate_encode(x, steps=12):
    x = torch.as_tensor(x, dtype=torch.float32)
    probability = torch.sigmoid(x)
    return torch.bernoulli(probability.unsqueeze(0).expand(steps, *probability.shape))
```

The random sequence changes on every call. Fix all random seeds when comparing experiments.

## 2.3 One-layer classifier

```python
import torch.nn as nn

class SpikeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad):
        return grad

class FirstSNN(nn.Module):
    def __init__(self, in_features=20, num_classes=5, beta=0.9):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.beta = beta

    def forward(self, spike_input):
        steps, batch, _ = spike_input.shape
        membrane = torch.zeros(batch, self.fc.out_features, device=spike_input.device)
        count = torch.zeros_like(membrane)
        for t in range(steps):
            membrane = self.beta * membrane + self.fc(spike_input[t])
            output_spike = SpikeSTE.apply(membrane - 1.0)
            membrane = membrane - output_spike
            count = count + output_spike
        return count / steps
```

The class with the largest output firing rate is the prediction.

## 2.4 Training loop

```python
model = FirstSNN().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

X_train_t = torch.tensor(X_train, dtype=torch.float32, device="cuda")
y_train_t = torch.tensor(y_train, dtype=torch.long, device="cuda")

for epoch in range(30):
    model.train()
    permutation = torch.randperm(len(X_train_t), device="cuda")
    for start in range(0, len(permutation), 128):
        index = permutation[start:start + 128]
        rates = model(rate_encode(X_train_t[index], steps=12))
        loss = criterion(rates, y_train_t[index])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch={epoch:02d} loss={loss.item():.4f}")
```

For CPU execution, remove `.cuda()` and use `device="cpu"`.

## 2.5 Common failure modes

- No spikes: lower the threshold, increase input scale, or inspect membrane values.
- Every neuron spikes: reduce the input scale or learning rate.
- Unstable accuracy: increase the number of steps or use deterministic constant-current encoding.
- Good training accuracy but poor test accuracy: add a validation split and regularization.
- CUDA device mismatch: create every state tensor on `x.device`.

## 2.6 Summary

The complete path is continuous features, spike encoding, LIF dynamics, spike-count logits, surrogate-gradient training, and argmax classification. The next chapter adds hidden layers, learnable decay, and a reliable validation procedure.

Next: [03-deeper-multistep.md](03-deeper-multistep.md)
