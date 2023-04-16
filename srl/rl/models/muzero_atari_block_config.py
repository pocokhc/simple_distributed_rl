from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from .base_block_config import IAlphaZeroImageBlockConfig


@dataclass
class MuzeroAtariBlockConfig(IAlphaZeroImageBlockConfig):
    base_filters: int = 128
    use_layer_normalization: bool = False
    kwargs: Dict[str, Any] = field(default_factory=lambda: {})

    def create_block_tf(self):
        from .tf import muzero_atari_block

        return muzero_atari_block.MuZeroAtariBlock(
            self.base_filters,
            use_layer_normalization=self.use_layer_normalization,
            **self.kwargs,
        )

    def create_block_torch(self, in_shape: Tuple[int, ...]):
        from .torch_ import muzero_atari_block

        return muzero_atari_block.MuZeroAtariBlock(
            in_shape,
            self.base_filters,
            use_layer_normalization=self.use_layer_normalization,
            **self.kwargs,
        )
