from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class MLPBlockConfig:
    _name: str = field(init=False, default="MLP")
    _kwargs: dict = field(init=False, default_factory=dict)

    def set(
        self,
        layer_sizes: Tuple[int, ...] = (512,),
        activation: str = "relu",
        **kwargs,
    ):
        """Multi-layer Perceptron Block

        Args:
            layer_sizes (Tuple[int, ...], optional): 各レイヤーのユニット数. Defaults to (512,).
            activation (str, optional): Activation function. Defaults to "relu".

        Examples:
            >>> mlp_conf = MLPBlockConfig()
            >>> mlp_conf.set((128, 64, 32))
        """
        self._name = "MLP"
        self._kwargs = dict(
            layer_sizes=layer_sizes,
            activation=activation.lower(),
        )
        self._kwargs.update(kwargs)
        return self

    def set_custom_block(self, entry_point: str, **kwargs):
        self._name = "custom"
        self._kwargs = dict(entry_point=entry_point, kwargs=kwargs)
        return self

    # ---------------------

    def create_block_tf(self):
        from srl.rl.tf.blocks.mlp_block import create_mlp_block_from_config

        return create_mlp_block_from_config(self)

    def create_block_torch(self, in_size: int):
        from srl.rl.torch_.blocks.mlp_block import create_mlp_block_from_config

        return create_mlp_block_from_config(self, in_size)
