import torch
import torch.nn as nn
from srl.rl.models.converter import convert_activation_torch


class DuelingNetworkBlock(nn.Module):
    def __init__(
        self,
        in_size: int,
        action_num: int,
        dense_units: int,
        dueling_type: str = "average",
        activation="ReLU",
        enable_noisy_dense: bool = False,
        enable_time_distributed_layer: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dueling_type = dueling_type

        assert len(layer_sizes) > 0

        activation = convert_activation_torch(activation)

        if enable_noisy_dense:
            raise NotImplementedError("TODO")
            _Dense = None
        else:
            _Dense = nn.Linear

        # value
        self.v1 = _Dense(in_size, dense_units)
        self.v1_act = activation(inplace=True)
        self.v2 = _Dense(dense_units, 1)

        # advance
        self.adv1 = _Dense(in_size, dense_units)
        self.adv1_act = activation(inplace=True)
        self.adv2 = _Dense(dense_units, action_num)

        if enable_time_distributed_layer:
            pass  # TODO

    def forward(self, x):
        v = self.v1(x)
        v = self.v1_act(v)
        v = self.v2(v)
        adv = self.adv1(x)
        adv = self.adv1_act(adv)
        adv = self.adv2(adv)

        dim = 1

        if self.dueling_type == "average":
            x = v + adv - torch.mean(adv, dim=dim, keepdim=True)
        elif self.dueling_type == "max":
            x = v + adv - torch.max(adv, dim=dim, keepdim=True)[0]
        elif self.dueling_type == "":  # naive
            x = v + adv
        else:
            raise ValueError("dueling_network_type is undefined")

        return x


if __name__ == "__main__":
    m = DuelingNetworkBlock(128, 5, 256)
    print(m)
