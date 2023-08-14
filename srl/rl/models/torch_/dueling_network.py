from typing import Tuple

import torch
import torch.nn as nn

from srl.rl.models.converter import convert_activation_torch
from srl.rl.models.torch_.noisy_linear import NoisyLinear


class DuelingNetworkBlock(nn.Module):
    def __init__(
        self,
        in_size: int,
        action_num: int,
        layer_sizes: Tuple[int, ...],
        dueling_type: str = "average",
        activation="ReLU",
        enable_time_distributed_layer: bool = False,
        enable_noisy_dense: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dueling_type = dueling_type
        self.enable_time_distributed_layer = enable_time_distributed_layer

        assert len(layer_sizes) > 0

        activation = convert_activation_torch(activation)

        if enable_noisy_dense:
            _Linear = NoisyLinear
        else:
            _Linear = nn.Linear

        # hidden
        self.hidden_layers = nn.ModuleList()
        if len(layer_sizes) > 1:
            for i in range(len(layer_sizes) - 1):
                self.hidden_layers.append(_Linear(in_size, layer_sizes[i]))
                self.hidden_layers.append(activation(inplace=True))
                in_size = layer_sizes[i]

        # value
        self.v_layers = nn.ModuleList()
        self.v_layers.append(_Linear(in_size, layer_sizes[-1]))
        self.v_layers.append(activation(inplace=True))
        self.v_layers.append(_Linear(layer_sizes[-1], 1))

        # advance
        self.adv_layers = nn.ModuleList()
        self.adv_layers.append(_Linear(in_size, layer_sizes[-1]))
        self.adv_layers.append(activation(inplace=True))
        self.adv_layers.append(_Linear(layer_sizes[-1], action_num))

    def forward(self, x):
        if self.enable_time_distributed_layer:
            # (batch, seq, units) -> (batch*seq, units)
            batch_size, seq, units = x.size()
            x = x.reshape((batch_size * seq, units))

        for layer in self.hidden_layers:
            x = layer(x)

        v = x
        for v_layer in self.v_layers:
            v = v_layer(v)

        adv = x
        for adv_layer in self.adv_layers:
            adv = adv_layer(adv)

        dim = 1

        if self.dueling_type == "average":
            x = v + adv - torch.mean(adv, dim=dim, keepdim=True)
        elif self.dueling_type == "max":
            x = v + adv - torch.max(adv, dim=dim, keepdim=True)[0]
        elif self.dueling_type == "":  # naive
            x = v + adv
        else:
            raise ValueError("dueling_network_type is undefined")

        if self.enable_time_distributed_layer:
            # (batch*seq, units) -> (batch, seq, units)
            _, units = x.size()
            x = x.view(batch_size, seq, units)

        return x


class NormalBlock(nn.Module):
    def __init__(
        self,
        in_size: int,
        action_num: int,
        layer_sizes: Tuple[int, ...],
        activation="ReLU",
        enable_time_distributed_layer: bool = False,
        enable_noisy_dense: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enable_time_distributed_layer = enable_time_distributed_layer

        assert len(layer_sizes) > 0

        activation = convert_activation_torch(activation)

        if enable_noisy_dense:
            _Linear = NoisyLinear
        else:
            _Linear = nn.Linear

        self.hidden_layers = nn.ModuleList()
        if len(layer_sizes) > 1:
            for i in range(len(layer_sizes) - 1):
                self.hidden_layers.append(_Linear(in_size, layer_sizes[i]))
                self.hidden_layers.append(activation(inplace=True))
                in_size = layer_sizes[i]

        self.hidden_layers.append(_Linear(in_size, layer_sizes[-1]))
        self.hidden_layers.append(activation(inplace=True))
        self.hidden_layers.append(_Linear(layer_sizes[-1], action_num))

    def forward(self, x):
        if self.enable_time_distributed_layer:
            # (batch, seq, units) -> (batch*seq, units)
            batch_size, seq, units = x.size()
            x = x.reshape((batch_size * seq, units))

            for layer in self.hidden_layers:
                x = layer(x)

            # (batch*seq, units) -> (batch, seq, units)
            _, units = x.size()
            x = x.view(batch_size, seq, units)

        else:
            for layer in self.hidden_layers:
                x = layer(x)

        return x


if __name__ == "__main__":
    # m = DuelingNetworkBlock(128, 5, (256,))
    m = NormalBlock(128, 5, (256,))
    print(m)
