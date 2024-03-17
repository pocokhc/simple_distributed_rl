from typing import Tuple

import torch
import torch.nn as nn

from srl.rl.models.torch_.converter import convert_activation_torch
from srl.rl.models.torch_.modules.noisy_linear import NoisyLinear


class DuelingNetworkBlock(nn.Module):
    def __init__(
        self,
        in_size: int,
        hidden_units: int,
        out_layer_units: int,
        dueling_type: str = "average",
        activation="ReLU",
        enable_noisy_dense: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dueling_type = dueling_type

        activation = convert_activation_torch(activation)

        if enable_noisy_dense:
            _Linear = NoisyLinear
        else:
            _Linear = nn.Linear

        # value
        self.v_layers = nn.ModuleList()
        self.v_layers.append(_Linear(in_size, hidden_units))
        self.v_layers.append(activation(inplace=True))
        self.v_layers.append(_Linear(hidden_units, 1))

        # advance
        self.adv_layers = nn.ModuleList()
        self.adv_layers.append(_Linear(in_size, hidden_units))
        self.adv_layers.append(activation(inplace=True))
        self.adv_layers.append(_Linear(hidden_units, out_layer_units))

    def forward(self, x):
        v = x
        for v_layer in self.v_layers:
            v = v_layer(v)

        adv = x
        for adv_layer in self.adv_layers:
            adv = adv_layer(adv)

        if self.dueling_type == "average":
            x = v + adv - torch.mean(adv, dim=-1, keepdim=True)
        elif self.dueling_type == "max":
            x = v + adv - torch.max(adv, dim=-1, keepdim=True)[0]
        elif self.dueling_type == "":  # naive
            x = v + adv
        else:
            raise ValueError("dueling_network_type is undefined")

        return x


if __name__ == "__main__":
    m = DuelingNetworkBlock(128, 5, 256)
    print(m)
