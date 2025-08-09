from dataclasses import dataclass, field
from typing import Literal, Tuple

from srl.base.exception import UndefinedError


@dataclass
class DuelingNetworkConfig:
    name: str = field(default="")
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.name == "":
            self.set_dueling_network((512,))

    def set(self, layer_sizes: Tuple[int, ...], activation: str = "relu", **kwargs):
        """Multi-layer Perceptron Block

        Args:
            layer_sizes (Tuple[int, ...], optional): 各レイヤーのユニット数. Defaults to (512,).
            activation (str, optional): Activation function. Defaults to "relu".

        Examples:
            >>> conf = DuelingNetworkConfig()
            >>> conf.set((128, 64, 32))
        """
        self.name = "MLP"
        self.kwargs = dict(
            layer_sizes=layer_sizes,
            activation=activation.lower(),
        )
        self.kwargs.update(kwargs)
        return self

    def set_dueling_network(
        self,
        layer_sizes: Tuple[int, ...],
        activation: str = "relu",
        dueling_type: Literal["", "average", "max"] = "average",
        **kwargs,
    ):
        """Multi-layer Perceptron Block + DuelingNetwork Block

        Args:
            layer_sizes (Tuple[int, ...], optional): 各レイヤーのユニット数. Defaults to (512,).
            activation (str, optional): Activation function. Defaults to "relu".
            dueling_type (str, optional): select algorithm. Defaults to "average".
        """

        self.name = "DuelingNetwork"
        mlp_kwargs = dict(activation=activation.lower())
        mlp_kwargs.update(kwargs)
        self.kwargs = dict(
            layer_sizes=layer_sizes,
            mlp_kwargs=mlp_kwargs,
            dueling_kwargs=dict(
                dueling_type=dueling_type,
                activation=activation.lower(),
            ),
        )
        return self

    def set_custom_block(self, entry_point: str, kwargs: dict):
        self.name = "custom"
        self.kwargs = dict(entry_point=entry_point, kwargs=kwargs)
        return self

    # ---------------------

    def create_tf_block(
        self,
        out_size: int,
        rnn: bool = False,
        enable_noisy_dense: bool = False,
        **kwargs,
    ):
        if self.name == "MLP":
            from tensorflow.keras import layers

            from srl.rl.tf.blocks.mlp_block import MLPBlock

            kwargs2 = self.kwargs.copy()
            kwargs2.update(kwargs)
            block = MLPBlock(enable_noisy_dense=enable_noisy_dense, **kwargs2)
            block.add_layer(layers.Dense(out_size, kernel_initializer="truncated_normal"))
            return block

        if self.name == "DuelingNetwork":
            from srl.rl.tf.blocks.dueling_network import DuelingNetworkBlock
            from srl.rl.tf.blocks.mlp_block import MLPBlock

            layer_sizes = self.kwargs["layer_sizes"]
            dueling_units = layer_sizes[-1]
            layer_sizes = layer_sizes[:-1]

            mlp_kwargs = self.kwargs["mlp_kwargs"]
            mlp_kwargs.update(kwargs)
            dueling_kwargs = self.kwargs["dueling_kwargs"]

            block = MLPBlock(layer_sizes, enable_noisy_dense=enable_noisy_dense, **mlp_kwargs)
            block.add_layer(
                DuelingNetworkBlock(
                    dueling_units,
                    out_size,
                    enable_noisy_dense=enable_noisy_dense,
                    **dueling_kwargs,
                )
            )
            return block

        if self.name == "custom":
            from srl.utils.common import load_module

            kwargs2 = self.kwargs["kwargs"].copy()
            kwargs2.update(kwargs)
            return load_module(self.kwargs["entry_point"])(out_size, rnn=rnn, **kwargs2)

        raise UndefinedError(self)

    def create_torch_block(self, in_size: int, out_size: int, enable_noisy_dense: bool = False):
        if self.name == "MLP":
            import torch.nn as nn

            from srl.rl.torch_.blocks.mlp_block import MLPBlock

            block = MLPBlock(in_size, enable_noisy_dense=enable_noisy_dense, **self.kwargs)
            block.add_layer(nn.Linear(block.out_size, out_size), out_size)
            return block

        if self.name == "DuelingNetwork":
            from srl.rl.torch_.blocks.dueling_network import DuelingNetworkBlock
            from srl.rl.torch_.blocks.mlp_block import MLPBlock

            layer_sizes = self.kwargs["layer_sizes"]
            dueling_units = layer_sizes[-1]
            layer_sizes = layer_sizes[:-1]

            block = MLPBlock(
                in_size,
                layer_sizes,
                enable_noisy_dense=enable_noisy_dense,
                **self.kwargs["mlp_kwargs"],
            )
            block.add_layer(
                DuelingNetworkBlock(
                    block.out_size,
                    dueling_units,
                    out_size,
                    enable_noisy_dense=enable_noisy_dense,
                    **self.kwargs["dueling_kwargs"],
                ),
                out_size,
            )
            return block

        if self.name == "custom":
            from srl.utils.common import load_module

            return load_module(self.kwargs["entry_point"])(in_size, out_size, **self.kwargs["kwargs"])

        raise UndefinedError(self)
