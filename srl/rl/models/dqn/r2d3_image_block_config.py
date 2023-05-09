from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from srl.base.rl.model import IImageBlockConfig


@dataclass
class R2D3ImageBlockConfig(IImageBlockConfig):
    kwargs: Dict[str, Any] = field(default_factory=lambda: {})

    def create_block_tf(self, enable_time_distributed_layer: bool = False):
        from .tf import r2d3_image_block

        return r2d3_image_block.R2D3ImageBlock(
            enable_time_distributed_layer=enable_time_distributed_layer,
            **self.kwargs,
        )

    def create_block_torch(self, in_shape: Tuple[int], enable_time_distributed_layer: bool = False):
        from .torch_ import r2d3_image_block

        return r2d3_image_block.R2D3ImageBlock(
            in_shape,
            enable_time_distributed_layer=enable_time_distributed_layer,
            **self.kwargs,
        )
