from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from .base_block_config import IMLPBlockConfig


@dataclass
class MLPBlockConfig(IMLPBlockConfig):
    layer_sizes: Tuple[int, ...] = (512,)
    kwargs: Dict[str, Any] = field(default_factory=lambda: {})

    def create_block_tf(self):
        from .tf import mlp_block

        return mlp_block.MLPBlock(
            self.layer_sizes,
            **self.kwargs,
        )

    def create_block_torch(self, in_size: int):
        from .torch_ import mlp_block

        return mlp_block.MLPBlock(
            in_size,
            self.layer_sizes,
            **self.kwargs,
        )
