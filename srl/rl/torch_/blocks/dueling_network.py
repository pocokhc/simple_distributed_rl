import torch
import torch.nn as nn

from srl.base.exception import UndefinedError
from srl.rl.models.config.dueling_network import DuelingNetworkConfig
from srl.rl.torch_.blocks.mlp_block import MLPBlock
from srl.rl.torch_.converter import convert_activation_torch
from srl.rl.torch_.modules.noisy_linear import NoisyLinear


def create_block_from_config(
    config: DuelingNetworkConfig,
    in_size: int,
    out_size: int,
    enable_noisy_dense: bool = False,
):
    if config.name == "MLP":
        block = MLPBlock(in_size, enable_noisy_dense=enable_noisy_dense, **config.kwargs)
        block.add_layer(nn.Linear(block.out_size, out_size), out_size)
        return block

    if config.name == "DuelingNetwork":
        layer_sizes = config.kwargs["layer_sizes"]
        dueling_units = layer_sizes[-1]
        layer_sizes = layer_sizes[:-1]

        block = MLPBlock(
            in_size,
            layer_sizes,
            enable_noisy_dense=enable_noisy_dense,
            **config.kwargs["mlp_kwargs"],
        )
        block.add_layer(
            DuelingNetworkBlock(
                block.out_size,
                dueling_units,
                out_size,
                enable_noisy_dense=enable_noisy_dense,
                **config.kwargs["dueling_kwargs"],
            ),
            out_size,
        )
        return block

    if config.name == "custom":
        from srl.utils.common import load_module

        return load_module(config.kwargs["entry_point"])(in_size, out_size, **config.kwargs["kwargs"])

    raise UndefinedError(config.name)


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
        self.v_layers.append(activation())
        self.v_layers.append(_Linear(hidden_units, 1))

        # advance
        self.adv_layers = nn.ModuleList()
        self.adv_layers.append(_Linear(in_size, hidden_units))
        self.adv_layers.append(activation())
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
