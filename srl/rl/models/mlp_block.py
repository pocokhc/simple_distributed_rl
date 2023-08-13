from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class MLPBlockConfig:
    _kwargs: Dict[str, Any] = field(init=False, default_factory=lambda: {"layer_sizes": (512,)})

    def set_mlp(
        self,
        layer_sizes: Tuple[int, ...] = (512,),
        activation: str = "relu",
        # kernel_initializer="he_normal", TODO
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
            activation=activation.lower(),
            # kernel_initializer=kernel_initializer,
        )

    def set_custom_block(self):
        """TODO"""
        raise NotImplementedError("TODO")

    # ---------------------

    def create_block_tf(self, enable_noisy_dense: bool = False):
        from .tf import mlp_block

        return mlp_block.MLPBlock(
            **self._kwargs,
            enable_time_distributed_layer=False,
            enable_noisy_dense=enable_noisy_dense,
        )

    def create_block_torch(self, in_size: int, enable_noisy_dense: bool = False):
        from .torch_ import mlp_block

        return mlp_block.MLPBlock(
            in_size,
            self._kwargs["layer_sizes"],
            # TODO 互換パラメータ
            enable_noisy_dense=enable_noisy_dense,
        )
