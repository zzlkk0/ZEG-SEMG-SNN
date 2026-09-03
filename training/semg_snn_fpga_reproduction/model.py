from __future__ import annotations

import torch
from torch import nn


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage_minus_threshold: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(voltage_minus_threshold)
        return (voltage_minus_threshold >= 0).to(voltage_minus_threshold.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        (x,) = ctx.saved_tensors
        slope = 10.0
        grad = 1.0 / (1.0 + slope * x.abs()).pow(2)
        return (grad_output * grad,)


spike_fn = SurrogateSpike.apply


class DenseLIFLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        decay: float = 0.9,
        threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.synapse = nn.Linear(in_features, out_features, bias=False)
        self.decay = decay
        self.threshold = threshold
        nn.init.xavier_uniform_(self.synapse.weight)

    def step(
        self, spikes: torch.Tensor, voltage: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        voltage_pre_reset = self.decay * voltage + self.synapse(spikes)
        output = spike_fn(voltage_pre_reset - self.threshold)
        voltage = voltage_pre_reset * (1.0 - output.detach())
        return output, voltage


class PaperSNN(nn.Module):
    """Dense 96 -> 64 -> 128 -> 64 -> 13 LIF topology."""

    def __init__(
        self,
        decay: float = 0.9,
        threshold: float = 1.0,
        classes: int = 13,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DenseLIFLayer(96, 64, decay, threshold),
                DenseLIFLayer(64, 128, decay, threshold),
                DenseLIFLayer(128, 64, decay, threshold),
                DenseLIFLayer(64, classes, decay, threshold),
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Args:
            x: [batch, time, 96] binary input spikes.

        Returns:
            output spikes [batch, time, classes] and per-layer spike rates.
        """
        batch, time, _ = x.shape
        voltages = [
            x.new_zeros((batch, layer.synapse.out_features))
            for layer in self.layers
        ]
        outputs = []
        layer_sums = [x.new_zeros(()) for _ in self.layers]

        for t in range(time):
            spikes = x[:, t]
            for index, layer in enumerate(self.layers):
                spikes, voltages[index] = layer.step(spikes, voltages[index])
                layer_sums[index] = layer_sums[index] + spikes.mean()
            outputs.append(spikes)

        output = torch.stack(outputs, dim=1)
        rates = [total / time for total in layer_sums]
        return output, rates


class LearnableAxonalDelay(nn.Module):
    """Per-output-neuron differentiable delay constrained to [0, max_delay].

    The forward path linearly interpolates adjacent integer delays. This is a
    practical pure-PyTorch approximation to SLAYER's learnable axonal delay.
    """

    def __init__(
        self, features: int, max_delay: int = 62, initial_delay: float = 1.0
    ) -> None:
        super().__init__()
        self.max_delay = int(max_delay)
        ratio = min(max(initial_delay / max_delay, 1e-4), 1.0 - 1e-4)
        initial_logit = torch.logit(torch.tensor(ratio))
        self.delay_logits = nn.Parameter(initial_logit.repeat(features))

    @property
    def delays(self) -> torch.Tensor:
        return self.max_delay * torch.sigmoid(self.delay_logits)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        # spikes: [batch, time, features]
        batch, time, features = spikes.shape
        delays = self.delays
        low = torch.floor(delays)
        fraction = delays - low
        low_index = low.to(torch.long)
        high_index = torch.clamp(low_index + 1, max=self.max_delay)
        time_index = torch.arange(time, device=spikes.device)[:, None]

        source_low = time_index - low_index[None, :]
        source_high = time_index - high_index[None, :]
        valid_low = source_low >= 0
        valid_high = source_high >= 0
        source_low = source_low.clamp(min=0)
        source_high = source_high.clamp(min=0)

        low_values = torch.gather(
            spikes, 1, source_low[None].expand(batch, -1, -1)
        )
        high_values = torch.gather(
            spikes, 1, source_high[None].expand(batch, -1, -1)
        )
        low_values = low_values * valid_low[None]
        high_values = high_values * valid_high[None]
        return (
            low_values * (1.0 - fraction)[None, None, :]
            + high_values * fraction[None, None, :]
        )


class PaperSNNWithDelays(nn.Module):
    """Paper topology with learnable axonal delays on the first three layers."""

    def __init__(
        self,
        decay: float = 0.9,
        threshold: float = 1.0,
        classes: int = 13,
        max_delay: int = 62,
        initial_delay: float = 1.0,
    ) -> None:
        super().__init__()
        dimensions = [(96, 64), (64, 128), (128, 64), (64, classes)]
        self.layers = nn.ModuleList(
            [
                DenseLIFLayer(in_features, out_features, decay, threshold)
                for in_features, out_features in dimensions
            ]
        )
        self.delays = nn.ModuleList(
            [
                LearnableAxonalDelay(
                    out_features, max_delay=max_delay, initial_delay=initial_delay
                )
                for _, out_features in dimensions[:3]
            ]
        )

    @staticmethod
    def run_lif_sequence(
        layer: DenseLIFLayer, inputs: torch.Tensor
    ) -> torch.Tensor:
        batch, time, _ = inputs.shape
        voltage = inputs.new_zeros((batch, layer.synapse.out_features))
        outputs = []
        for t in range(time):
            spikes, voltage = layer.step(inputs[:, t], voltage)
            outputs.append(spikes)
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        rates = []
        for index, layer in enumerate(self.layers):
            x = self.run_lif_sequence(layer, x)
            rates.append(x.mean())
            if index < len(self.delays):
                x = self.delays[index](x)
        return x, rates

    def delay_statistics(self) -> list[dict[str, float]]:
        output = []
        for delay in self.delays:
            values = delay.delays.detach()
            output.append(
                {
                    "minimum": float(values.min()),
                    "mean": float(values.mean()),
                    "maximum": float(values.max()),
                }
            )
        return output


def spike_rate_loss(
    output_spikes: torch.Tensor,
    targets: torch.Tensor,
    true_rate: float = 0.2,
    false_rate: float = 0.03,
) -> torch.Tensor:
    rates = output_spikes.mean(dim=1)
    desired = torch.full_like(rates, false_rate)
    desired.scatter_(1, targets[:, None], true_rate)
    return nn.functional.mse_loss(rates, desired)
