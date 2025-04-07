from dataclasses import dataclass, field
from typing import Tuple

from srl.base.spaces.space import SpaceBase


@dataclass
class InputValueBlockConfig:
    name: str = field(default="")
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.name == "":
            self.set()

    def set(
        self,
        layer_sizes: Tuple[int, ...] = (),
        activation: str = "relu",
        **kwargs,
    ):
        """Multi-layer Perceptron Block

        Args:
            layer_sizes (Tuple[int, ...], optional): 各レイヤーのユニット数. Defaults to ().
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

    def create_tf_block(self, in_space: SpaceBase, input_flatten: bool = True, rnn: bool = False):
        from srl.rl.tf.blocks.input_value_block import create_block_from_config

        return create_block_from_config(self, in_space, input_flatten, rnn)

    def create_torch_block(self, in_shape: tuple, input_flatten: bool = True, reshape_for_rnn: bool = False):
        from srl.rl.torch_.blocks.input_value_block import create_block_from_config

        return create_block_from_config(self, in_shape, input_flatten, reshape_for_rnn)
