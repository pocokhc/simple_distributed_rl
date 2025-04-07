from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class MLPBlockConfig:
    name: str = field(default="")
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.name == "":
            self.set()

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
        self.name = "MLP"
        self.kwargs = dict(
            layer_sizes=layer_sizes,
            activation=activation.lower(),
        )
        self.kwargs.update(kwargs)
        return self

    def set_custom_block(self, entry_point: str, **kwargs):
        self.name = "custom"
        self.kwargs = dict(entry_point=entry_point, kwargs=kwargs)
        return self

    # ---------------------

    def create_tf_block(self):
        from srl.rl.tf.blocks.mlp_block import create_block_from_config

        return create_block_from_config(self)

    def create_torch_block(self, in_size: int):
        from srl.rl.torch_.blocks.mlp_block import create_block_from_config

        return create_block_from_config(self, in_size)
