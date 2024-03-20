import logging
from dataclasses import dataclass
from typing import Tuple

from srl.base.exception import UndefinedError

logger = logging.getLogger(__name__)


@dataclass
class MLPBlockConfig:
    def __post_init__(self):
        self._name: str = ""
        self._kwargs: dict = {}

        self.set()

    def set(
        self,
        layer_sizes: Tuple[int, ...] = (512,),
        activation: str = "relu",
        use_bias=True,
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        # kernel_regularizer=None,
        # bias_regularizer=None,
        # activity_regularizer=None,
        # kernel_constraint=None,
        # bias_constraint=None,
    ):
        """Multi-layer Perceptron Block

        Args:
            layer_sizes (Tuple[int, ...], optional): 各レイヤーのユニット数. Defaults to (512,).
            activation (str, optional): Activation function. Defaults to "relu".

        Examples:
            >>> mlp_conf = MLPBlockConfig()
            >>> mlp_conf.set_mlp((128, 64, 32))
        """
        self._name = "MLP"
        self._kwargs = dict(
            layer_sizes=layer_sizes,
            activation=activation.lower(),
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
        return self

    def set_dueling_network(
        self,
        layer_sizes: Tuple[int, ...] = (512,),
        activation: str = "relu",
        use_bias=True,
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        # kernel_regularizer=None,
        # bias_regularizer=None,
        # activity_regularizer=None,
        # kernel_constraint=None,
        # bias_constraint=None,
        dueling_type: str = "average",
    ):
        """Multi-layer Perceptron Block + DuelingNetwork Block

        Args:
            layer_sizes (Tuple[int, ...], optional): 各レイヤーのユニット数. Defaults to (512,).
            activation (str, optional): Activation function. Defaults to "relu".
            dueling_type (str, optional): select algorithm. Defaults to "average".

        """

        self._name = "DuelingNetwork"
        self._kwargs = dict(
            layer_sizes=layer_sizes,
            mlp_kwargs=dict(
                activation=activation.lower(),
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            ),
            dueling_kwargs=dict(
                dueling_type=dueling_type,
                activation=activation.lower(),
            ),
        )
        return self

    def set_custom_block(self, entry_point: str, kwargs: dict):
        self._name = "custom"
        self._kwargs = dict(entry_point=entry_point, kwargs=kwargs)
        return self

    # ---------------------

    def create_block_tf(
        self,
        out_units: int = -1,
        enable_rnn: bool = False,
        enable_noisy_dense: bool = False,
    ):
        if self._name == "MLP":
            from tensorflow import keras

            from srl.rl.models.tf.blocks.mlp_block import MLPBlock

            block = MLPBlock(enable_noisy_dense=enable_noisy_dense, **self._kwargs)
            if out_units > 0:
                block.add_layer(
                    keras.layers.Dense(
                        out_units,
                        kernel_initializer="truncated_normal",
                        bias_initializer="truncated_normal",
                    )
                )

            return block

        if self._name == "DuelingNetwork":
            from srl.rl.models.tf.blocks.dueling_network import DuelingNetworkBlock
            from srl.rl.models.tf.blocks.mlp_block import MLPBlock

            layer_sizes = self._kwargs["layer_sizes"]
            if out_units > 0:
                dueling_units = layer_sizes[-1]
                layer_sizes = layer_sizes[:-1]
            else:
                logger.warning("Dueling Network cannot be used")

            block = MLPBlock(
                layer_sizes,
                enable_noisy_dense=enable_noisy_dense,
                **self._kwargs["mlp_kwargs"],
            )
            if out_units > 0:
                block.add_layer(
                    DuelingNetworkBlock(
                        dueling_units,
                        out_units,
                        enable_noisy_dense=enable_noisy_dense,
                        **self._kwargs["dueling_kwargs"],
                    )
                )

            return block

        if self._name == "custom":
            from srl.utils.common import load_module

            return load_module(self._kwargs["entry_point"])(enable_rnn=enable_rnn, **self._kwargs["kwargs"])

        raise UndefinedError(self._name)

    def create_block_torch(
        self,
        in_size: int,
        out_units: int = -1,
        enable_noisy_dense: bool = False,
    ):
        if self._name == "MLP":
            import torch.nn as nn

            from srl.rl.models.torch_.blocks.mlp_block import MLPBlock

            block = MLPBlock(in_size, enable_noisy_dense=enable_noisy_dense, **self._kwargs)
            if out_units > 0:
                block.add_layer(nn.Linear(block.out_size, out_units), out_units)

            return block

        if self._name == "DuelingNetwork":
            from srl.rl.models.torch_.blocks.dueling_network import DuelingNetworkBlock
            from srl.rl.models.torch_.blocks.mlp_block import MLPBlock

            layer_sizes = self._kwargs["layer_sizes"]
            if out_units > 0:
                dueling_units = layer_sizes[-1]
                layer_sizes = layer_sizes[:-1]
            else:
                logger.warning("Dueling Network cannot be used")

            block = MLPBlock(
                in_size,
                layer_sizes,
                enable_noisy_dense=enable_noisy_dense,
                **self._kwargs["mlp_kwargs"],
            )
            if out_units > 0:
                block.add_layer(
                    DuelingNetworkBlock(
                        block.out_size,
                        dueling_units,
                        out_units,
                        enable_noisy_dense=enable_noisy_dense,
                        **self._kwargs["dueling_kwargs"],
                    ),
                    out_units,
                )

            return block

        if self._name == "custom":
            from srl.utils.common import load_module

            return load_module(self._kwargs["entry_point"])(**self._kwargs["kwargs"])

        raise UndefinedError(self._name)
