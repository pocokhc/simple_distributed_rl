from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from srl.base.exception import UndefinedError


@dataclass
class MLPBlockConfig:
    _name: str = field(init=False, default="mlp")
    _kwargs: Dict[str, Any] = field(init=False, default_factory=lambda: {"layer_sizes": (512,)})

    def set_mlp(
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
        self._name = "mlp"
        self._kwargs = dict(
            layer_sizes=layer_sizes,
            activation=activation.lower(),
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )

    def set_custom_block(self, entry_point: str, kwargs: dict):
        self._name = "custom"
        self._kwargs = dict(
            entry_point=entry_point,
            kwargs=kwargs,
        )

    # ---------------------

    def create_block_tf(self):
        if self._name == "mlp":
            from .tf import mlp_block

            return mlp_block.MLPBlock(**self._kwargs)

        if self._name == "custom":
            from srl.utils.common import load_module

            return load_module(self._kwargs["entry_point"])(**self._kwargs["kwargs"])

        raise UndefinedError(self._name)

    def create_block_torch(self, in_size: int):
        if self._name == "mlp":
            from .torch_ import mlp_block

            return mlp_block.MLPBlock(in_size, **self._kwargs)

        if self._name == "custom":
            from srl.utils.common import load_module

            return load_module(self._kwargs["entry_point"])(**self._kwargs["kwargs"])

        raise UndefinedError(self._name)
