from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class AlphaZeroBlockConfig:
    _name: str = field(init=False, default_factory=str)
    _kwargs: Dict[str, Any] = field(init=False, default_factory=lambda: {})

    def set_alphazero_block(
        self,
        n_blocks: int = 19,
        filters: int = 256,
    ):
        self._name = "AlphaZero"
        self._kwargs = dict(
            n_blocks=n_blocks,
            filters=filters,
        )

    def set_muzero_atari_block(
        self,
        filters: int = 128,
        use_layer_normalization: bool = False,
    ):
        self._name = "MuzeroAtari"
        self._kwargs = dict(
            filters=filters,
            use_layer_normalization=use_layer_normalization,
        )

    def set_original_block(self):
        raise NotImplementedError("TODO")

    # ---------------------

    def create_block_tf(self):
        if self._name == "AlphaZero":
            from .tf.alphazero_image_block import AlphaZeroImageBlock

            return AlphaZeroImageBlock(**self._kwargs)
        elif self._name == "MuzeroAtari":
            from .tf.muzero_atari_block import MuZeroAtariBlock

            return MuZeroAtariBlock(**self._kwargs)
        else:
            raise ValueError(self._name)

    def create_block_torch(self, in_shape: Tuple[int, ...]):
        if self._name == "AlphaZero":
            from .torch_.alphazero_image_block import AlphaZeroImageBlock

            return AlphaZeroImageBlock(in_shape, **self._kwargs)
        elif self._name == "MuzeroAtari":
            from .torch_.muzero_atari_block import MuZeroAtariBlock

            return MuZeroAtariBlock(in_shape, **self._kwargs)
        else:
            raise ValueError(self._name)
