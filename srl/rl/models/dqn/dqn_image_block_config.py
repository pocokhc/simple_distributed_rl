from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from srl.base.rl.model import IImageBlockConfig


@dataclass
class DQNImageBlockConfig(IImageBlockConfig):
    filter: int = 32
    kwargs: Dict[str, Any] = field(default_factory=lambda: {})

    def create_block_tf(self, enable_time_distributed_layer: bool = False):
        from .tf import dqn_image_block

        return dqn_image_block.DQNImageBlock(
            self.filter,
            enable_time_distributed_layer=enable_time_distributed_layer,
            **self.kwargs,
        )

    def create_block_torch(
        self,
        in_shape: Tuple[int, ...],
        enable_time_distributed_layer: bool = False,
    ):
        from .torch_ import dqn_image_block

        return dqn_image_block.DQNImageBlock(
            in_shape,
            self.filter,
            enable_time_distributed_layer=enable_time_distributed_layer,
            **self.kwargs,
        )
