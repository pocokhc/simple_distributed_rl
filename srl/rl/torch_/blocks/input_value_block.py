from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from srl.base.exception import UndefinedError
from srl.rl.models.config.input_value_block import InputValueBlockConfig
from srl.rl.torch_.blocks.a_input_block import AInputBlock
from srl.rl.torch_.converter import apply_initializer_torch, convert_activation_torch
from srl.rl.torch_.modules.noisy_linear import NoisyLinear


def create_block_from_config(config: InputValueBlockConfig, in_shape: tuple, input_flatten: bool, reshape_for_rnn: bool):
    if config.name == "MLP":
        return InputValueBlock(in_shape, input_flatten=input_flatten, reshape_for_rnn=reshape_for_rnn, **config.kwargs)

    if config.name == "custom":
        from srl.utils.common import load_module

        return load_module(config.kwargs["entry_point"])(in_shape, input_flatten=input_flatten, reshape_for_rnn=reshape_for_rnn, **config.kwargs["kwargs"])

    raise UndefinedError(config.name)


class InputValueBlock(nn.Module, AInputBlock):
    def __init__(
        self,
        in_shape: Tuple[int, ...],
        layer_sizes: Tuple[int, ...] = (),
        activation="ReLU",
        use_bias: bool = True,
        kernel_initializer: str = "he_normal",
        bias_initializer: str = "zeros",
        enable_noisy_dense: bool = False,
        input_flatten: bool = True,
        reshape_for_rnn: bool = False,
    ):
        super().__init__()
        self.reshape_for_rnn = reshape_for_rnn

        activation = convert_activation_torch(activation)
        self.hidden_layers = nn.ModuleList()

        if input_flatten:
            self.hidden_layers.append(nn.Flatten())
            in_size = int(np.prod(in_shape))
        else:
            in_size = in_shape[-1]

        for size in layer_sizes:
            if enable_noisy_dense:
                layer = NoisyLinear(in_size, size)
            else:
                layer = nn.Linear(in_size, size, bias=use_bias)
                if kernel_initializer != "":
                    apply_initializer_torch(layer.weight, kernel_initializer)
                if use_bias and bias_initializer != "":
                    apply_initializer_torch(layer.bias, bias_initializer)
            self.hidden_layers.append(layer)
            self.hidden_layers.append(activation(inplace=True))
            in_size = size

        self.out_size = in_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reshape_for_rnn:
            self.batch_size, self.seq_len, *feat_dims = x.size()
            x = x.view((self.batch_size * self.seq_len, *feat_dims))
        for layer in self.hidden_layers:
            x = layer(x)
        return x

    def unreshape_for_rnn(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self.batch_size, self.seq_len, *x.shape[1:])
