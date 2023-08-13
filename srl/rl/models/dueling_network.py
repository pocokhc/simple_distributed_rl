from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class DuelingNetworkConfig:
    _kwargs: Dict[str, Any] = field(init=False, default_factory=lambda: {})
    _enable: bool = False

    def set(
        self,
        layer_sizes: Tuple[int, ...],
        enable: bool,
        dueling_type: str = "average",
        activation: str = "relu",
    ):
        """Multi-layer Perceptron Block + DuelingNetwork Block

        Args:
            layer_sizes (Tuple[int, ...], optional): 各レイヤーのユニット数。ただし最後の層はDuelingNetwork用です。 Defaults to (512,).
            enable (bool): 無効にした場合、ただのMLP層になります。
            dueling_type (str, optional): select algorithm. Defaults to "average".
            activation (str, optional): Activation function. Defaults to "relu".
        """
        self._enable = enable
        self._kwargs = dict(
            layer_sizes=layer_sizes,
            dueling_type=dueling_type,
            activation=activation,
        )

    # ---------------------

    def create_block_tf(
        self,
        action_num: int,
        enable_noisy_dense: bool = False,
        enable_time_distributed_layer: bool = False,
    ):
        if self._enable:
            from .tf import dueling_network

            return dueling_network.DuelingNetworkBlock(
                action_num=action_num,
                **self._kwargs,
                enable_noisy_dense=enable_noisy_dense,
                enable_time_distributed_layer=enable_time_distributed_layer,
            )
        else:
            from .tf import dueling_network

            d = self._kwargs.copy()
            del d["dueling_type"]

            return dueling_network.NormalBlock(
                action_num=action_num,
                **d,
                enable_noisy_dense=enable_noisy_dense,
                enable_time_distributed_layer=enable_time_distributed_layer,
            )

    def create_block_torch(
        self,
        in_size: int,
        action_num: int,
        enable_noisy_dense: bool = False,
        enable_time_distributed_layer: bool = False,
    ):
        if self._enable:
            from .torch_ import dueling_network

            return dueling_network.DuelingNetworkBlock(
                in_size,
                action_num=action_num,
                **self._kwargs,
                enable_noisy_dense=enable_noisy_dense,
                enable_time_distributed_layer=enable_time_distributed_layer,
            )
        else:
            from .torch_ import dueling_network

            d = self._kwargs.copy()
            del d["dueling_type"]

            return dueling_network.NormalBlock(
                in_size,
                action_num=action_num,
                **d,
                enable_noisy_dense=enable_noisy_dense,
                enable_time_distributed_layer=enable_time_distributed_layer,
            )
