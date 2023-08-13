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
        activation: str = "relu",
    ):
        """Alphaシリーズの画像レイヤーで使用する層を指定します。
        AlphaZeroで採用されている層です。

        Args:
            n_blocks (int, optional): ブロック数. Defaults to 19.
            filters (int, optional): フィルター数. Defaults to 256.
            activation (str, optional): activation function. Defaults to "relu".
        """
        self._name = "AlphaZero"
        self._kwargs = dict(
            n_blocks=n_blocks,
            filters=filters,
            activation=activation,
        )

    def set_muzero_atari_block(
        self,
        filters: int = 128,
        activation: str = "relu",
        use_layer_normalization: bool = False,
    ):
        """Alphaシリーズの画像レイヤーで使用する層を指定します。
        MuZeroのAtari環境で採用されている層です。

        Args:
            filters (int, optional): フィルター数. Defaults to 128.
            activation (str, optional): activation function. Defaults to "relu".
            use_layer_normalization (str, optional): use_layer_normalization. Defaults to True.
        """
        self._name = "MuzeroAtari"
        self._kwargs = dict(
            filters=filters,
            activation=activation,
            use_layer_normalization=use_layer_normalization,
        )

    def set_original_block(self):
        """TODO"""
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
