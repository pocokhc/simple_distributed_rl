from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class MLPBlockConfig:
    _kwargs: Dict[str, Any] = field(init=False, default_factory=lambda: {"layer_sizes": (512,)})

    def set_mlp(
        self,
        layer_sizes: Tuple[int, ...] = (512,),
        activation="relu",
        kernel_initializer="he_normal",
        dense_kwargs: Dict[str, Any] = {},
    ):
        """Multi-layer Perceptron Block

        Args:
            layer_sizes (Tuple[int, ...], optional): 各レイヤーのユニット数. Defaults to (512,).
            activation (str, optional): Activation function. Defaults to "relu".

        Examples:
            >>> mlp_conf = MLPBlockConfig()
            >>> mlp_conf.set_mlp((128, 64, 32))
        """
        self._kwargs = dict(
            layer_sizes=layer_sizes,
            activation=activation,
            kernel_initializer=kernel_initializer,
            dense_kwargs=dense_kwargs,
        )

    def set_custom_block(self):
        """TODO"""
        raise NotImplementedError("TODO")

    # ---------------------

    def create_block_tf(self):
        from .tf import mlp_block

        return mlp_block.MLPBlock(
            **self._kwargs,
            enable_time_distributed_layer=False,
        )

    def create_block_torch(self, in_size: int):
        from .torch_ import mlp_block

        return mlp_block.MLPBlock(
            in_size,
            self._kwargs["layer_sizes"],
            # **self._kwargs,  TODO 互換パラメータの作成
        )
