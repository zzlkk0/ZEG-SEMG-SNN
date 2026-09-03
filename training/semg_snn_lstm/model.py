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
        return (grad_output / (1.0 + slope * x.abs()).pow(2),)


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
        self, inputs: torch.Tensor, voltage: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        voltage_pre_reset = self.decay * voltage + self.synapse(inputs)
        spikes = spike_fn(voltage_pre_reset - self.threshold)
        voltage_after_reset = voltage_pre_reset * (1.0 - spikes.detach())
        return spikes, voltage_pre_reset, voltage_after_reset


class SNNLSTM(nn.Module):
    """96 -> 64 LIF -> 128 LIF membrane sequence -> LSTM(64) -> 13."""

    def __init__(
        self,
        classes: int = 13,
        decay: float = 0.9,
        threshold: float = 1.0,
        lstm_hidden: int = 64,
        dropout: float = 0.1,
        feature_mode: str = "membrane",
    ) -> None:
        super().__init__()
        if feature_mode not in {"membrane", "spike", "both"}:
            raise ValueError(feature_mode)
        self.feature_mode = feature_mode
        self.lif1 = DenseLIFLayer(96, 64, decay, threshold)
        self.lif2 = DenseLIFLayer(64, 128, decay, threshold)
        lstm_input = 256 if feature_mode == "both" else 128
        self.feature_norm = nn.LayerNorm(lstm_input)
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden, classes)

    def load_snn_frontend(self, checkpoint: dict) -> None:
        state = checkpoint["model"]
        self.lif1.synapse.weight.data.copy_(state["layers.0.synapse.weight"])
        self.lif2.synapse.weight.data.copy_(state["layers.1.synapse.weight"])

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, time, _ = x.shape
        voltage1 = x.new_zeros((batch, 64))
        voltage2 = x.new_zeros((batch, 128))
        features = []
        spike1_sum = x.new_zeros(())
        spike2_sum = x.new_zeros(())

        for t in range(time):
            spike1, _, voltage1 = self.lif1.step(x[:, t], voltage1)
            spike2, voltage2_pre, voltage2 = self.lif2.step(spike1, voltage2)
            spike1_sum = spike1_sum + spike1.mean()
            spike2_sum = spike2_sum + spike2.mean()

            if self.feature_mode == "membrane":
                feature = voltage2_pre
            elif self.feature_mode == "spike":
                feature = spike2
            else:
                feature = torch.cat((voltage2_pre, spike2), dim=-1)
            features.append(feature)

        sequence = self.feature_norm(torch.stack(features, dim=1))
        lstm_output, _ = self.lstm(sequence)
        pooled = lstm_output.mean(dim=1)
        logits = self.classifier(self.dropout(pooled))
        diagnostics = {
            "lif1_spike_rate": spike1_sum / time,
            "lif2_spike_rate": spike2_sum / time,
        }
        return logits, diagnostics


class ResidualSNNLSTM(nn.Module):
    """Frozen paper SNN plus an LSTM residual correction branch.

    The model starts with exactly the paper-SNN class ranking. The zero-initialized
    LSTM head can only add corrections as it learns, avoiding destructive
    fine-tuning of the already strong SNN classifier.
    """

    def __init__(
        self,
        classes: int = 13,
        decay: float = 0.9,
        threshold: float = 1.0,
        lstm_hidden: int = 64,
        dropout: float = 0.1,
        snn_scale: float = 20.0,
    ) -> None:
        super().__init__()
        dimensions = [(96, 64), (64, 128), (128, 64), (64, classes)]
        self.snn_layers = nn.ModuleList(
            [
                DenseLIFLayer(in_features, out_features, decay, threshold)
                for in_features, out_features in dimensions
            ]
        )
        self.feature_norm = nn.LayerNorm(128)
        self.lstm = nn.LSTM(128, lstm_hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.residual_head = nn.Linear(lstm_hidden, classes)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.snn_scale = float(snn_scale)

    def load_snn_frontend(self, checkpoint: dict) -> None:
        state = checkpoint["model"]
        for index, layer in enumerate(self.snn_layers):
            layer.synapse.weight.data.copy_(
                state[f"layers.{index}.synapse.weight"]
            )

    def freeze_snn(self) -> None:
        for layer in self.snn_layers:
            for parameter in layer.parameters():
                parameter.requires_grad = False

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, time, _ = x.shape
        voltages = [
            x.new_zeros((batch, layer.synapse.out_features))
            for layer in self.snn_layers
        ]
        membrane_features = []
        output_spikes = []
        spike_sums = [x.new_zeros(()) for _ in self.snn_layers]

        for t in range(time):
            spikes = x[:, t]
            for index, layer in enumerate(self.snn_layers):
                spikes, voltage_pre, voltages[index] = layer.step(
                    spikes, voltages[index]
                )
                spike_sums[index] = spike_sums[index] + spikes.mean()
                if index == 1:
                    membrane_features.append(voltage_pre)
            output_spikes.append(spikes)

        base_rates = torch.stack(output_spikes, dim=1).mean(dim=1)
        sequence = self.feature_norm(torch.stack(membrane_features, dim=1))
        lstm_output, _ = self.lstm(sequence)
        residual = self.residual_head(
            self.dropout(lstm_output.mean(dim=1))
        )
        logits = self.snn_scale * base_rates + residual
        return logits, {
            "lif1_spike_rate": spike_sums[0] / time,
            "lif2_spike_rate": spike_sums[1] / time,
        }


class LearnableAxonalDelay(nn.Module):
    def __init__(self, features: int, max_delay: int = 62) -> None:
        super().__init__()
        self.max_delay = max_delay
        self.delay_logits = nn.Parameter(torch.zeros(features))

    @property
    def delays(self) -> torch.Tensor:
        return self.max_delay * torch.sigmoid(self.delay_logits)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        batch, time, _ = spikes.shape
        delays = self.delays
        low = torch.floor(delays)
        fraction = delays - low
        low_index = low.long()
        high_index = torch.clamp(low_index + 1, max=self.max_delay)
        time_index = torch.arange(time, device=spikes.device)[:, None]
        source_low = time_index - low_index[None]
        source_high = time_index - high_index[None]
        valid_low = source_low >= 0
        valid_high = source_high >= 0
        low_values = torch.gather(
            spikes,
            1,
            source_low.clamp(min=0)[None].expand(batch, -1, -1),
        ) * valid_low[None]
        high_values = torch.gather(
            spikes,
            1,
            source_high.clamp(min=0)[None].expand(batch, -1, -1),
        ) * valid_high[None]
        return (
            low_values * (1.0 - fraction)[None, None]
            + high_values * fraction[None, None]
        )


class DelayedResidualSNNLSTM(nn.Module):
    """Frozen learned-delay SNN with an LSTM residual correction branch."""

    def __init__(
        self,
        classes: int = 13,
        decay: float = 0.9,
        threshold: float = 1.0,
        lstm_hidden: int = 64,
        dropout: float = 0.1,
        snn_scale: float = 20.0,
        max_delay: int = 62,
    ) -> None:
        super().__init__()
        dimensions = [(96, 64), (64, 128), (128, 64), (64, classes)]
        self.snn_layers = nn.ModuleList(
            [
                DenseLIFLayer(in_features, out_features, decay, threshold)
                for in_features, out_features in dimensions
            ]
        )
        self.delays = nn.ModuleList(
            [
                LearnableAxonalDelay(out_features, max_delay)
                for _, out_features in dimensions[:3]
            ]
        )
        self.feature_norm = nn.LayerNorm(128)
        self.lstm = nn.LSTM(128, lstm_hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.residual_head = nn.Linear(lstm_hidden, classes)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.snn_scale = snn_scale

    def load_snn_frontend(self, checkpoint: dict) -> None:
        state = checkpoint["model"]
        for index, layer in enumerate(self.snn_layers):
            layer.synapse.weight.data.copy_(
                state[f"layers.{index}.synapse.weight"]
            )
        for index, delay in enumerate(self.delays):
            delay.delay_logits.data.copy_(state[f"delays.{index}.delay_logits"])

    def freeze_snn(self) -> None:
        for layer in self.snn_layers:
            for parameter in layer.parameters():
                parameter.requires_grad = False
        for delay in self.delays:
            for parameter in delay.parameters():
                parameter.requires_grad = False

    @staticmethod
    def run_layer(
        layer: DenseLIFLayer, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, time, _ = inputs.shape
        voltage = inputs.new_zeros((batch, layer.synapse.out_features))
        spikes_out, membrane_out = [], []
        for t in range(time):
            spikes, membrane_pre, voltage = layer.step(inputs[:, t], voltage)
            spikes_out.append(spikes)
            membrane_out.append(membrane_pre)
        return torch.stack(spikes_out, dim=1), torch.stack(membrane_out, dim=1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rates = []
        layer2_membrane = None
        for index, layer in enumerate(self.snn_layers):
            x, membrane = self.run_layer(layer, x)
            rates.append(x.mean())
            if index == 1:
                layer2_membrane = membrane
            if index < len(self.delays):
                x = self.delays[index](x)

        base_rates = x.mean(dim=1)
        sequence = self.feature_norm(layer2_membrane)
        lstm_output, _ = self.lstm(sequence)
        residual = self.residual_head(
            self.dropout(lstm_output.mean(dim=1))
        )
        return self.snn_scale * base_rates + residual, {
            "lif1_spike_rate": rates[0],
            "lif2_spike_rate": rates[1],
        }
