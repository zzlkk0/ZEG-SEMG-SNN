from __future__ import annotations

import math

import torch
from torch import nn

from model import spike


class GradedRF(nn.Module):
    """RF bank using phase-zero crossings and graded real-state events."""

    def __init__(self, neurons: int = 80, fs: float = 200.0):
        super().__init__()
        frequencies = torch.linspace(20.0, 100.0, neurons)
        decay = torch.linspace(0.00774, 0.0933, neurons)
        # The paper reports the threshold range [2, .02]. Higher frequencies
        # use the smaller threshold because their larger decay reduces state.
        threshold = torch.linspace(2.0, 0.02, neurons)
        angle = 2 * math.pi * frequencies / fs
        self.register_buffer("cos", angle.cos()[None, None])
        self.register_buffer("sin", angle.sin()[None, None])
        self.register_buffer("retention", (1.0 - decay)[None, None])
        self.register_buffer("threshold", threshold[None, None])
        self.neurons = neurons

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # DB5 is signed 8-bit. This is equivalent to applying the paper's
        # fixed-point 1<<12 scale and then returning to Q12 for computation.
        x = x.float() / 128.0
        batch, time, channels = x.shape
        real = x.new_zeros(batch, channels, self.neurons)
        imag = torch.zeros_like(real)
        previous_imag = torch.zeros_like(real)
        events = []
        active = x.new_zeros(())
        for t in range(time):
            old_real = real
            real = self.retention * (old_real * self.cos - imag * self.sin) + x[:, t, :, None]
            imag = self.retention * (old_real * self.sin + imag * self.cos)
            crossed = (previous_imag * imag <= 0) & (previous_imag != imag)
            graded = torch.where(crossed & (real >= self.threshold), real, torch.zeros_like(real))
            # Negative polarity gets its own event stream, preserving signed EMG.
            graded_neg = torch.where(crossed & (-real >= self.threshold), -real, torch.zeros_like(real))
            event = torch.cat((graded, graded_neg), dim=-1).flatten(1)
            events.append(event)
            active += event.gt(0).float().mean()
            previous_imag = imag
        return torch.stack(events, dim=1), active / time


class FaithfulRFSNN(nn.Module):
    def __init__(self, rf_neurons: int = 80, hidden1: int = 256, hidden2: int = 128):
        super().__init__()
        self.rf = GradedRF(rf_neurons)
        self.fc1 = nn.Linear(16 * rf_neurons * 2, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, 13)
        self.norm1 = nn.LayerNorm(hidden1)
        self.norm2 = nn.LayerNorm(hidden2)
        self.beta1_logit = nn.Parameter(torch.tensor(2.2))
        self.beta2_logit = nn.Parameter(torch.tensor(2.2))

    def forward(self, x: torch.Tensor):
        rf, rf_rate = self.rf(x)
        mem1 = x.new_zeros(x.shape[0], self.fc1.out_features)
        mem2 = x.new_zeros(x.shape[0], self.fc2.out_features)
        output_mem = x.new_zeros(x.shape[0], 13)
        count1 = x.new_zeros(())
        count2 = x.new_zeros(())
        beta1 = torch.sigmoid(self.beta1_logit)
        beta2 = torch.sigmoid(self.beta2_logit)
        for t in range(rf.shape[1]):
            cur1 = self.norm1(self.fc1(torch.log1p(rf[:, t])))
            mem1 = beta1 * mem1 + cur1
            spk1 = spike(mem1 - 1.0)
            mem1 = mem1 - spk1
            mem2 = beta2 * mem2 + self.norm2(self.fc2(spk1))
            spk2 = spike(mem2 - 1.0)
            mem2 = mem2 - spk2
            output_mem = output_mem + self.out(spk2)
            count1 += spk1.mean()
            count2 += spk2.mean()
        return output_mem / rf.shape[1], [rf_rate, count1 / rf.shape[1], count2 / rf.shape[1]]
