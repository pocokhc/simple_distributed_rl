from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from .base_block_config import IAlphaZeroImageBlockConfig


@dataclass
class AlphaZeroImageBlockConfig(IAlphaZeroImageBlockConfig):
    n_blocks: int = 19
    filters: int = 256
    use_layer_normalization: bool = False
    kwargs: Dict[str, Any] = field(default_factory=lambda: {})

    def create_block_tf(self):
        from .tf import alphazero_image_block

        return alphazero_image_block.AlphaZeroImageBlock(
            self.n_blocks,
            self.filters,
            use_layer_normalization=self.use_layer_normalization,
            **self.kwargs,
        )

    def create_block_torch(self, in_shape: Tuple[int, ...]):
        from .torch_ import alphazero_image_block

        return alphazero_image_block.AlphaZeroImageBlock(
            in_shape,
            self.n_blocks,
            self.filters,
            use_layer_normalization=self.use_layer_normalization,
            **self.kwargs,
        )
