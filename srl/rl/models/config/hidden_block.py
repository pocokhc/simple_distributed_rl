from dataclasses import dataclass, field
from typing import Tuple

from srl.base.exception import UndefinedError


@dataclass
class HiddenBlockConfig:
    name: str = field(default="")
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.name == "":
            self.set((512,))

    def set(self, layer_sizes: Tuple[int, ...], activation: str = "relu", **kwargs):
        """Multi-layer Perceptron Block

        Args:
            layer_sizes (Tuple[int, ...], optional): 各レイヤーのユニット数. Defaults to (512,).
            activation (str, optional): Activation function. Defaults to "relu".

        Examples:
            >>> conf = HiddenBlockConfig()
            >>> conf.set((128, 64, 32))
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

    def create_tf_block(self, **kwargs):
        if self.name == "MLP":
            from srl.rl.tf.blocks.mlp_block import MLPBlock

            kwargs2 = self.kwargs.copy()
            kwargs2.update(kwargs)
            return MLPBlock(**kwargs2)
        elif self.name == "custom":
            from srl.utils.common import load_module

            kwargs2 = self.kwargs["kwargs"].copy()
            kwargs2.update(kwargs)
            return load_module(self.kwargs["entry_point"])(**kwargs2)
        else:
            raise UndefinedError(self)

    def create_torch_block(self, in_size: int):
        if self.name == "MLP":
            from srl.rl.torch_.blocks.mlp_block import MLPBlock

            return MLPBlock(in_size, **self.kwargs)

        if self.name == "custom":
            from srl.utils.common import load_module

            return load_module(self.kwargs["entry_point"])(in_size, **self.kwargs["kwargs"])

        raise UndefinedError(self)
