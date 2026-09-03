# 03. Deeper Multi-Step SNNs

The single-layer classifier is useful for learning, but its capacity is limited. This chapter adds two LIF layers, learnable decay, substeps, gradient clipping, and validation-based early stopping.

## 3.1 Deterministic constant-current encoding

Instead of Bernoulli sampling, repeat the normalized feature vector at every step and let recurrent membrane dynamics create the temporal behavior:

```python
def constant_encode(x, steps):
    """[batch, features] -> [steps, batch, features]."""
    x = torch.as_tensor(x, dtype=torch.float32)
    return x.unsqueeze(0).expand(steps, *x.shape).contiguous()
```

This removes input-sampling noise and is often more stable for fixed feature windows.

## 3.2 Learnable decay

```python
import torch.nn as nn

class LearnableBeta(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(channels))

    def value(self):
        return torch.sigmoid(self.raw)
```

Using one value per neuron lets the network learn multiple time constants. On hardware, however, it also requires more stored decay values than a shared scalar.

## 3.3 Two-layer model

```python
class SpikeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad):
        return grad

spike = SpikeSTE.apply

class TwoLayerSNN(nn.Module):
    def __init__(self, in_features, hidden1, hidden2, num_classes, substeps=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, num_classes)
        self.beta1 = LearnableBeta(hidden1)
        self.beta2 = LearnableBeta(hidden2)
        self.substeps = substeps

    def forward(self, x_seq):
        steps, batch, _ = x_seq.shape
        m1 = x_seq.new_zeros(batch, self.fc1.out_features)
        m2 = x_seq.new_zeros(batch, self.fc2.out_features)
        output_sum = x_seq.new_zeros(batch, self.out.out_features)
        for t in range(steps):
            current1 = self.fc1(x_seq[t])
            for _ in range(self.substeps):
                m1 = self.beta1.value() * m1 + current1
                s1 = spike(m1 - 1.0)
                m1 = m1 - s1
                m2 = self.beta2.value() * m2 + self.fc2(s1)
                s2 = spike(m2 - 1.0)
                m2 = m2 - s2
                output_sum = output_sum + self.out(s2)
        return output_sum / (steps * self.substeps)
```

Substeps provide additional integration cycles per input frame without changing the external encoding length.

## 3.4 Robust training

Use AdamW, gradient clipping, a learning-rate schedule, and early stopping selected only by validation accuracy:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
best_val, best_state, bad_epochs, patience = 0.0, None, 0, 8

for epoch in range(40):
    model.train()
    for x_batch, y_batch in train_loader:
        logits = model(constant_encode(x_batch, steps=8))
        loss = torch.nn.functional.cross_entropy(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
    scheduler.step()

    val_acc = evaluate(model, validation_loader)
    if val_acc > best_val:
        best_val = val_acc
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        bad_epochs = 0
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            break
```

Never select a checkpoint or tune a hyperparameter by test accuracy. Doing so indirectly overfits the test set.

## 3.5 Soft reset and hard reset

- Soft reset subtracts the threshold and preserves excess membrane charge. It is often easier to train.
- Hard reset sets the membrane to zero after a spike. It can simplify hardware but may lose useful state.

Treat reset style as a model-and-hardware design choice and report it explicitly.

## 3.6 Summary

The model now has learnable time constants, multiple integration cycles, gradient control, and honest checkpoint selection. The next chapter combines models that observe different representations.

Next: [04-parallel-branches-fusion.md](04-parallel-branches-fusion.md)
