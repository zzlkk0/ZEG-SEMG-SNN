from __future__ import annotations

import math

import torch
from torch import nn


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return grad / (1.0 + 5.0 * x.abs()).pow(2)


spike = Spike.apply


class RFFeatureBank(nn.Module):
    """Damped quadrature resonators approximating graded RF spikes."""

    def __init__(self, channels: int = 16, neurons: int = 80, fs: float = 200.0):
        super().__init__()
        frequencies = torch.linspace(20.0, 100.0, neurons)
        # Paper range is reported as decay [0.00774, 0.0933]. Interpret it as
        # loss per step; higher-frequency resonators receive stronger loss.
        loss = torch.linspace(0.00774, 0.0933, neurons)
        self.register_buffer("rotation", 2 * math.pi * frequencies / fs)
        self.register_buffer("retention", 1.0 - loss)
        self.channels = channels
        self.neurons = neurons

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        real = x.new_zeros(batch, self.channels, self.neurons)
        imag = x.new_zeros(batch, self.channels, self.neurons)
        cos = self.rotation.cos()[None, None]
        sin = self.rotation.sin()[None, None]
        retain = self.retention[None, None]
        positive = x.new_zeros(real.shape)
        negative = x.new_zeros(real.shape)
        activity = x.new_zeros(())
        for t in range(x.shape[1]):
            old_real = real
            real = retain * (old_real * cos - imag * sin) + x[:, t, :, None]
            imag = retain * (old_real * sin + imag * cos)
            # Graded RF events retain magnitude, with polarity in separate streams.
            gate = (imag.abs() < 0.35).to(x.dtype)
            pos = torch.relu(real - 0.15) * gate
            neg = torch.relu(-real - 0.15) * gate
            positive = positive + pos
            negative = negative + neg
            activity = activity + (pos.gt(0).float().mean() + neg.gt(0).float().mean()) / 2
        features = torch.cat((positive, negative), dim=-1).flatten(1) / x.shape[1]
        return torch.log1p(features), activity / x.shape[1]


class RFSNN(nn.Module):
    def __init__(self, hidden1: int = 384, hidden2: int = 192, steps: int = 12):
        super().__init__()
        self.rf = RFFeatureBank()
        self.fc1 = nn.Linear(16 * 80 * 2, hidden1)
        self.norm1 = nn.LayerNorm(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.norm2 = nn.LayerNorm(hidden2)
        self.out = nn.Linear(hidden2, 13)
        self.steps = steps
        self.decay1 = nn.Parameter(torch.tensor(1.8))
        self.decay2 = nn.Parameter(torch.tensor(1.8))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features, rf_rate = self.rf(x)
        current1 = self.norm1(self.fc1(features))
        mem1 = torch.zeros_like(current1)
        mem2 = x.new_zeros(x.shape[0], self.fc2.out_features)
        logits = x.new_zeros(x.shape[0], 13)
        rates1 = x.new_zeros(())
        rates2 = x.new_zeros(())
        beta1 = torch.sigmoid(self.decay1)
        beta2 = torch.sigmoid(self.decay2)
        for _ in range(self.steps):
            mem1 = beta1 * mem1 + current1
            spk1 = spike(mem1 - 1.0)
            mem1 = mem1 - spk1
            mem2 = beta2 * mem2 + self.norm2(self.fc2(spk1))
            spk2 = spike(mem2 - 1.0)
            mem2 = mem2 - spk2
            logits = logits + self.out(spk2)
            rates1 = rates1 + spk1.mean()
            rates2 = rates2 + spk2.mean()
        return logits / self.steps, [rf_rate, rates1 / self.steps, rates2 / self.steps]
