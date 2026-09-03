from __future__ import annotations

import torch
from torch import nn


class Surrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (1.0 + 5.0 * x.abs()).pow(2)


spike = Surrogate.apply


class FeatureSNN(nn.Module):
    """Rate-coded feature input with a LIF classifier core."""

    def __init__(self, features: int = 336, hidden1: int = 512, hidden2: int = 256,
                 steps: int = 16, dropout: float = 0.15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(features, hidden1),
            nn.LayerNorm(hidden1),
        )
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.norm2 = nn.LayerNorm(hidden2)
        self.out = nn.Linear(hidden2, 13)
        self.steps = steps
        self.dropout = nn.Dropout(dropout)
        self.beta1 = nn.Parameter(torch.tensor(2.2))
        self.beta2 = nn.Parameter(torch.tensor(2.2))

    def forward(self, features, raw=None, subject=None):
        current = self.encoder(features)
        m1 = torch.zeros_like(current)
        m2 = current.new_zeros(current.shape[0], self.fc2.out_features)
        output = current.new_zeros(current.shape[0], 13)
        r1 = current.new_zeros(())
        r2 = current.new_zeros(())
        b1, b2 = torch.sigmoid(self.beta1), torch.sigmoid(self.beta2)
        for _ in range(self.steps):
            m1 = b1 * m1 + current
            s1 = self.dropout(spike(m1 - 1.0))
            m1 = m1 - (s1 > 0).to(m1.dtype).detach()
            m2 = b2 * m2 + self.norm2(self.fc2(s1))
            s2 = self.dropout(spike(m2 - 1.0))
            m2 = m2 - (s2 > 0).to(m2.dtype).detach()
            output = output + self.out(s2)
            r1 += (s1 > 0).float().mean()
            r2 += (s2 > 0).float().mean()
        return output / self.steps, [r1 / self.steps, r2 / self.steps]


class SubjectFeatureSNN(FeatureSNN):
    """Shared spiking representation with calibrated per-subject readouts."""

    def __init__(self, features: int = 336, hidden1: int = 512, hidden2: int = 256,
                 steps: int = 16, dropout: float = 0.15):
        super().__init__(features, hidden1, hidden2, steps, dropout)
        del self.out
        self.out_weight = nn.Parameter(torch.empty(10, 13, hidden2))
        self.out_bias = nn.Parameter(torch.zeros(10, 13))
        nn.init.xavier_uniform_(self.out_weight)

    def forward(self, features, raw=None, subject=None):
        if subject is None:
            raise ValueError("subject ids are required for subject-conditioned heads")
        current = self.encoder(features)
        m1 = torch.zeros_like(current)
        m2 = current.new_zeros(current.shape[0], self.fc2.out_features)
        output = current.new_zeros(current.shape[0], 13)
        r1 = current.new_zeros(())
        r2 = current.new_zeros(())
        b1, b2 = torch.sigmoid(self.beta1), torch.sigmoid(self.beta2)
        weight, bias = self.out_weight[subject], self.out_bias[subject]
        for _ in range(self.steps):
            m1 = b1 * m1 + current
            s1 = self.dropout(spike(m1 - 1.0))
            m1 = m1 - (s1 > 0).to(m1.dtype).detach()
            m2 = b2 * m2 + self.norm2(self.fc2(s1))
            s2 = self.dropout(spike(m2 - 1.0))
            m2 = m2 - (s2 > 0).to(m2.dtype).detach()
            output = output + torch.bmm(weight, s2.unsqueeze(-1)).squeeze(-1) + bias
            r1 += (s1 > 0).float().mean()
            r2 += (s2 > 0).float().mean()
        return output / self.steps, [r1 / self.steps, r2 / self.steps]


class ContextSNN(FeatureSNN):
    """Causal ConvLIF-style SNN over consecutive feature windows."""

    def __init__(self, features: int = 336, hidden1: int = 512, hidden2: int = 256,
                 substeps: int = 3, dropout: float = 0.15):
        super().__init__(features, hidden1, hidden2, steps=substeps, dropout=dropout)
        self.substeps = substeps

    def forward(self, features, raw=None, subject=None):
        if features.ndim == 2:
            features = features[:, None]
        m1 = features.new_zeros(features.shape[0], self.encoder[0].out_features)
        m2 = features.new_zeros(features.shape[0], self.fc2.out_features)
        output = features.new_zeros(features.shape[0], 13)
        r1 = features.new_zeros(())
        r2 = features.new_zeros(())
        b1, b2 = torch.sigmoid(self.beta1), torch.sigmoid(self.beta2)
        total_steps = features.shape[1] * self.substeps
        for window in range(features.shape[1]):
            current = self.encoder(features[:, window])
            for _ in range(self.substeps):
                m1 = b1 * m1 + current
                s1 = self.dropout(spike(m1 - 1.0))
                m1 = m1 - (s1 > 0).to(m1.dtype).detach()
                m2 = b2 * m2 + self.norm2(self.fc2(s1))
                s2 = self.dropout(spike(m2 - 1.0))
                m2 = m2 - (s2 > 0).to(m2.dtype).detach()
                # Later evidence receives more weight without future context.
                recency = (window + 1) / features.shape[1]
                output = output + recency * self.out(s2)
                r1 += (s1 > 0).float().mean()
                r2 += (s2 > 0).float().mean()
        scale = self.substeps * sum((i + 1) / features.shape[1] for i in range(features.shape[1]))
        return output / scale, [r1 / total_steps, r2 / total_steps]


class SubjectFiLMContextSNN(ContextSNN):
    """Context SNN with small subject-specific modulation in the spiking core."""

    def __init__(self, features: int = 336, hidden1: int = 512, hidden2: int = 256,
                 substeps: int = 3, dropout: float = 0.15):
        super().__init__(features, hidden1, hidden2, substeps, dropout)
        self.film_scale = nn.Parameter(torch.zeros(10, hidden2))
        self.film_bias = nn.Parameter(torch.zeros(10, hidden2))

    def forward(self, features, raw=None, subject=None):
        if subject is None:
            raise ValueError("subject ids are required")
        if features.ndim == 2:
            features = features[:, None]
        m1 = features.new_zeros(features.shape[0], self.encoder[0].out_features)
        m2 = features.new_zeros(features.shape[0], self.fc2.out_features)
        output = features.new_zeros(features.shape[0], 13)
        r1 = features.new_zeros(())
        r2 = features.new_zeros(())
        b1, b2 = torch.sigmoid(self.beta1), torch.sigmoid(self.beta2)
        scale = 1.0 + 0.2 * torch.tanh(self.film_scale[subject])
        bias = 0.2 * self.film_bias[subject]
        total_steps = features.shape[1] * self.substeps
        for window in range(features.shape[1]):
            current = self.encoder(features[:, window])
            for _ in range(self.substeps):
                m1 = b1 * m1 + current
                s1 = self.dropout(spike(m1 - 1.0))
                m1 = m1 - (s1 > 0).to(m1.dtype).detach()
                current2 = self.norm2(self.fc2(s1)) * scale + bias
                m2 = b2 * m2 + current2
                s2 = self.dropout(spike(m2 - 1.0))
                m2 = m2 - (s2 > 0).to(m2.dtype).detach()
                recency = (window + 1) / features.shape[1]
                output = output + recency * self.out(s2)
                r1 += (s1 > 0).float().mean()
                r2 += (s2 > 0).float().mean()
        normalization = self.substeps * sum(
            (i + 1) / features.shape[1] for i in range(features.shape[1])
        )
        return output / normalization, [r1 / total_steps, r2 / total_steps]


class AdaptiveContextSNN(ContextSNN):
    """Context SNN with a learned causal exponential readout timescale."""

    def __init__(self, features: int = 336, hidden1: int = 512, hidden2: int = 256,
                 substeps: int = 3, dropout: float = 0.15):
        super().__init__(features, hidden1, hidden2, substeps, dropout)
        self.context_gamma_logit = nn.Parameter(torch.tensor(1.4))

    def forward(self, features, raw=None, subject=None):
        if features.ndim == 2:
            features = features[:, None]
        m1 = features.new_zeros(features.shape[0], self.encoder[0].out_features)
        m2 = features.new_zeros(features.shape[0], self.fc2.out_features)
        output = features.new_zeros(features.shape[0], 13)
        r1 = features.new_zeros(())
        r2 = features.new_zeros(())
        b1, b2 = torch.sigmoid(self.beta1), torch.sigmoid(self.beta2)
        gamma = torch.sigmoid(self.context_gamma_logit)
        weights = gamma ** torch.arange(
            features.shape[1] - 1, -1, -1, device=features.device
        )
        total_steps = features.shape[1] * self.substeps
        for window in range(features.shape[1]):
            current = self.encoder(features[:, window])
            for _ in range(self.substeps):
                m1 = b1 * m1 + current
                s1 = self.dropout(spike(m1 - 1.0))
                m1 = m1 - (s1 > 0).to(m1.dtype).detach()
                m2 = b2 * m2 + self.norm2(self.fc2(s1))
                s2 = self.dropout(spike(m2 - 1.0))
                m2 = m2 - (s2 > 0).to(m2.dtype).detach()
                output = output + weights[window] * self.out(s2)
                r1 += (s1 > 0).float().mean()
                r2 += (s2 > 0).float().mean()
        normalization = self.substeps * weights.sum()
        return output / normalization, [r1 / total_steps, r2 / total_steps, gamma]


class ClassAdaptiveContextSNN(ContextSNN):
    """PLIF context SNN with neuron- and class-specific causal timescales."""

    def __init__(self, features: int = 336, hidden1: int = 512, hidden2: int = 256,
                 substeps: int = 3, dropout: float = 0.15):
        super().__init__(features, hidden1, hidden2, substeps, dropout)
        # Per-class readout decay separates short transients from sustained/rest states.
        self.context_gamma_logit = nn.Parameter(torch.full((13,), 1.4))
        # Small heterogeneous offsets retain compatibility with the trained scalar PLIF
        # parameters while allowing individual neurons to learn distinct memories.
        self.beta1_offset = nn.Parameter(torch.zeros(hidden1))
        self.beta2_offset = nn.Parameter(torch.zeros(hidden2))

    def forward(self, features, raw=None, subject=None):
        if features.ndim == 2:
            features = features[:, None]
        m1 = features.new_zeros(features.shape[0], self.encoder[0].out_features)
        m2 = features.new_zeros(features.shape[0], self.fc2.out_features)
        output = features.new_zeros(features.shape[0], 13)
        r1 = features.new_zeros(())
        r2 = features.new_zeros(())
        b1 = torch.sigmoid(self.beta1 + 0.5 * torch.tanh(self.beta1_offset))[None]
        b2 = torch.sigmoid(self.beta2 + 0.5 * torch.tanh(self.beta2_offset))[None]
        gamma = torch.sigmoid(self.context_gamma_logit)
        lags = torch.arange(
            features.shape[1] - 1, -1, -1, device=features.device
        )
        weights = gamma[None].pow(lags[:, None])
        total_steps = features.shape[1] * self.substeps
        for window in range(features.shape[1]):
            current = self.encoder(features[:, window])
            for _ in range(self.substeps):
                m1 = b1 * m1 + current
                s1 = self.dropout(spike(m1 - 1.0))
                m1 = m1 - (s1 > 0).to(m1.dtype).detach()
                m2 = b2 * m2 + self.norm2(self.fc2(s1))
                s2 = self.dropout(spike(m2 - 1.0))
                m2 = m2 - (s2 > 0).to(m2.dtype).detach()
                output = output + weights[window] * self.out(s2)
                r1 += (s1 > 0).float().mean()
                r2 += (s2 > 0).float().mean()
        normalization = self.substeps * weights.sum(dim=0)
        return output / normalization, [
            r1 / total_steps, r2 / total_steps, gamma.mean(),
            gamma.min(), gamma.max(),
        ]


class ConvLIFBranch(nn.Module):
    def __init__(self, channels=128, mean_max=False):
        super().__init__()
        self.mean_max = mean_max
        self.front = nn.Sequential(
            nn.Conv1d(16, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, channels, 5, padding=2),
            nn.BatchNorm1d(channels),
        )
        self.q = nn.Conv1d(channels, channels, 1)
        self.k = nn.Conv1d(channels, channels, 1)
        self.v = nn.Conv1d(channels, channels, 1)
        self.beta = nn.Parameter(torch.tensor(2.2))

    def forward(self, raw):
        current = self.front(raw.transpose(1, 2))
        mem = torch.zeros_like(current[:, :, 0])
        spikes = []
        beta = torch.sigmoid(self.beta)
        for t in range(current.shape[-1]):
            mem = beta * mem + current[:, :, t]
            s = spike(mem - 1.0)
            mem = mem - s.detach()
            spikes.append(s)
        s = torch.stack(spikes, dim=-1)
        q, k, v = spike(self.q(s) - 0.5), spike(self.k(s) - 0.5), spike(self.v(s) - 0.5)
        intersection = torch.minimum(q, k).sum(dim=-1)
        union = torch.maximum(q, k).sum(dim=-1).clamp_min(1.0)
        attention = (intersection / union).unsqueeze(-1)
        attended = v * attention + s
        pooled = attended.mean(dim=-1)
        if self.mean_max:
            pooled = torch.cat((pooled, attended.amax(dim=-1)), dim=1)
        return pooled, [s.mean(), q.mean(), v.mean()]


class ContextHybridSNN(ClassAdaptiveContextSNN):
    """Continuous PLIF context with a residual ConvLIF/Jaccard raw branch."""

    def __init__(self, features: int = 336, hidden1: int = 512, hidden2: int = 256,
                 substeps: int = 3, dropout: float = 0.15, conv_channels: int = 128,
                 raw_steps: int = 8):
        super().__init__(features, hidden1, hidden2, substeps, dropout)
        self.conv = ConvLIFBranch(conv_channels)
        self.raw_current = nn.Sequential(
            nn.Linear(conv_channels, 128),
            nn.LayerNorm(128),
        )
        self.raw_out = nn.Linear(128, 13)
        self.beta_raw = nn.Parameter(torch.tensor(2.2))
        self.raw_scale_logit = nn.Parameter(torch.tensor(-1.4))
        self.raw_steps = raw_steps

    def forward(self, features, raw=None, subject=None):
        context_logits, rates = super().forward(features, raw, subject)
        conv, conv_rates = self.conv(raw)
        current = self.raw_current(conv)
        membrane = torch.zeros_like(current)
        raw_logits = current.new_zeros(current.shape[0], 13)
        raw_rate = current.new_zeros(())
        beta = torch.sigmoid(self.beta_raw)
        for _ in range(self.raw_steps):
            membrane = beta * membrane + current
            spikes = spike(membrane - 1.0)
            membrane = membrane - spikes.detach()
            raw_logits += self.raw_out(spikes)
            raw_rate += spikes.mean()
        scale = torch.sigmoid(self.raw_scale_logit)
        logits = context_logits + scale * raw_logits / self.raw_steps
        return logits, [*rates, raw_rate / self.raw_steps, scale, *conv_rates]


class HybridSNN(nn.Module):
    def __init__(self, features=336, hidden=384, conv_channels=128, steps=12,
                 mean_max=False):
        super().__init__()
        self.feature_current = nn.Sequential(nn.Linear(features, hidden), nn.LayerNorm(hidden))
        self.conv = ConvLIFBranch(conv_channels, mean_max=mean_max)
        conv_output = conv_channels * (2 if mean_max else 1)
        self.fuse = nn.Sequential(nn.Linear(hidden + conv_output, 256), nn.LayerNorm(256))
        self.out = nn.Linear(256, 13)
        self.steps = steps
        self.beta_f = nn.Parameter(torch.tensor(2.2))
        self.beta_o = nn.Parameter(torch.tensor(2.2))

    def forward(self, features, raw, subject=None):
        f_current = self.feature_current(features)
        conv, conv_rates = self.conv(raw)
        mf = torch.zeros_like(f_current)
        mo = f_current.new_zeros(f_current.shape[0], 256)
        logits = f_current.new_zeros(f_current.shape[0], 13)
        rf = f_current.new_zeros(())
        ro = f_current.new_zeros(())
        bf, bo = torch.sigmoid(self.beta_f), torch.sigmoid(self.beta_o)
        for _ in range(self.steps):
            mf = bf * mf + f_current
            sf = spike(mf - 1.0)
            mf = mf - sf.detach()
            mo = bo * mo + self.fuse(torch.cat((sf, conv), dim=1))
            so = spike(mo - 1.0)
            mo = mo - so.detach()
            logits += self.out(so)
            rf += sf.mean()
            ro += so.mean()
        return logits / self.steps, [rf / self.steps, ro / self.steps, *conv_rates]


class HybridMaxSNN(HybridSNN):
    def __init__(self, features=336, hidden=384, conv_channels=128, steps=12):
        super().__init__(features, hidden, conv_channels, steps, mean_max=True)


class SpatialConvLIFBranch(nn.Module):
    """Spiking spatial branch over the two 8-electrode Myo rings."""

    def __init__(self, channels=64):
        super().__init__()
        self.front = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, channels, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(channels),
        )
        self.beta = nn.Parameter(torch.tensor(2.2))

    def forward(self, raw):
        # [B,T,16] -> [B, two rings, eight electrodes, T]
        x = raw.transpose(1, 2).reshape(raw.shape[0], 2, 8, raw.shape[1])
        current = self.front(x).mean(dim=2)
        mem = current.new_zeros(current.shape[0], current.shape[1])
        spikes = []
        beta = torch.sigmoid(self.beta)
        for t in range(current.shape[-1]):
            mem = beta * mem + current[:, :, t]
            s = spike(mem - 1.0)
            mem = mem - s.detach()
            spikes.append(s)
        s = torch.stack(spikes, dim=-1)
        return torch.cat((s.mean(dim=-1), s.amax(dim=-1)), dim=1), s.mean()


class HybridSpatialSNN(nn.Module):
    def __init__(self, features=336, hidden=384, temporal_channels=128,
                 spatial_channels=64, steps=12):
        super().__init__()
        self.feature_current = nn.Sequential(nn.Linear(features, hidden), nn.LayerNorm(hidden))
        self.conv = ConvLIFBranch(temporal_channels)
        self.spatial = SpatialConvLIFBranch(spatial_channels)
        self.fuse = nn.Sequential(
            nn.Linear(hidden + temporal_channels + spatial_channels * 2, 256),
            nn.LayerNorm(256),
        )
        self.out = nn.Linear(256, 13)
        self.steps = steps
        self.beta_f = nn.Parameter(torch.tensor(2.2))
        self.beta_o = nn.Parameter(torch.tensor(2.2))

    def forward(self, features, raw, subject=None):
        f_current = self.feature_current(features)
        temporal, temporal_rates = self.conv(raw)
        spatial, spatial_rate = self.spatial(raw)
        mf = torch.zeros_like(f_current)
        mo = f_current.new_zeros(f_current.shape[0], 256)
        logits = f_current.new_zeros(f_current.shape[0], 13)
        rf = f_current.new_zeros(())
        ro = f_current.new_zeros(())
        bf, bo = torch.sigmoid(self.beta_f), torch.sigmoid(self.beta_o)
        for _ in range(self.steps):
            mf = bf * mf + f_current
            sf = spike(mf - 1.0)
            mf = mf - sf.detach()
            mo = bo * mo + self.fuse(torch.cat((sf, temporal, spatial), dim=1))
            so = spike(mo - 1.0)
            mo = mo - so.detach()
            logits += self.out(so)
            rf += sf.mean()
            ro += so.mean()
        return logits / self.steps, [
            rf / self.steps, ro / self.steps, *temporal_rates, spatial_rate
        ]
