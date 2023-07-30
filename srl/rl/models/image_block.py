from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class ImageBlockConfig:
    _name: str = field(init=False, default_factory=str)
    _kwargs: Dict[str, Any] = field(init=False, default_factory=lambda: {})

    def set_dqn_image(
        self,
        filters: int = 32,
        activation="relu",
    ):
        self._name = "DQN"
        self._kwargs = dict(
            filters=filters,
            activation=activation,
        )

    def set_r2d3_image(
        self,
        filters: int = 16,
    ):
        self._name = "R2D3"
        self._kwargs = dict(
            filters=filters,
        )

    # ---------------------

    def create_block_tf(self, enable_time_distributed_layer: bool):
        if self._name == "DQN":
            from .dqn.tf import dqn_image_block

            return dqn_image_block.DQNImageBlock(
                enable_time_distributed_layer=enable_time_distributed_layer,
                **self._kwargs,
            )
        elif self._name == "R2D3":
            from .dqn.tf import r2d3_image_block

            return r2d3_image_block.R2D3ImageBlock(
                enable_time_distributed_layer=enable_time_distributed_layer,
                **self._kwargs,
            )
        else:
            raise ValueError(self._name)

    def create_block_torch(
        self,
        in_shape: Tuple[int, ...],
        enable_time_distributed_layer: bool,
    ):
        if self._name == "DQN":
            from .dqn.torch_ import dqn_image_block

            return dqn_image_block.DQNImageBlock(
                in_shape,
                enable_time_distributed_layer=enable_time_distributed_layer,
                filters=self._kwargs["filters"],
            )
        elif self._name == "R2D3":
            from .dqn.torch_ import r2d3_image_block

            return r2d3_image_block.R2D3ImageBlock(
                in_shape,
                enable_time_distributed_layer=enable_time_distributed_layer,
                **self._kwargs,
            )
        else:
            raise ValueError(self._name)
