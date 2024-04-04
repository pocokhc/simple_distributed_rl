from dataclasses import dataclass
from typing import Optional, Tuple

from srl.base.define import SpaceTypes
from srl.base.exception import UndefinedError
from srl.base.rl.processor import ObservationProcessor
from srl.rl.processors.image_processor import ImageProcessor


@dataclass
class ImageBlockConfig:
    def __post_init__(self):
        self._name: str = ""
        self._kwargs: dict = {}
        self._processor: Optional[ObservationProcessor] = None

        self.set_dqn_block()

    def set_dqn_block(
        self,
        image_type: SpaceTypes = SpaceTypes.GRAY_2ch,
        resize: Tuple[int, int] = (84, 84),
        filters: int = 32,
        activation: str = "relu",
    ):
        """画像の入力に対してDQNで採用されたLayersを使用します。

        Args:
            image_type (SpaceTypes): 画像のタイプ. Defaults to SpaceTypes.GRAY_2ch
            resize (Tuple[int, int]): 画像のサイズ. Defaults to (84, 84)
            filters (int): 基準となるfilterの数です. Defaults to 32.
            activation (str): activation function. Defaults to "relu".
        """
        self._name = "DQN"
        self._kwargs = dict(filters=filters, activation=activation)
        self._processor = ImageProcessor(image_type, resize, enable_norm=True)
        return self

    def set_r2d3_block(
        self,
        image_type: SpaceTypes = SpaceTypes.COLOR,
        resize: Tuple[int, int] = (96, 72),
        filters: int = 16,
        activation: str = "relu",
    ):
        """画像の入力に対してR2D3で採用されたLayersを使用します。

        Args:
            image_type (SpaceTypes): 画像のタイプ. Defaults to SpaceTypes.COLOR
            resize (Tuple[int, int]): 画像のサイズ. Defaults to (96, 72)
            filters (int, optional): 基準となるfilterの数です. Defaults to 32.
            activation (str, optional): activation function. Defaults to "relu".
        """
        self._name = "R2D3"
        self._kwargs = dict(filters=filters, activation=activation)
        self._processor = ImageProcessor(image_type, resize, enable_norm=True)
        return self

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
        self._processor = None
        return self

    def set_muzero_atari_block(
        self,
        image_type: SpaceTypes = SpaceTypes.GRAY_2ch,
        resize: Tuple[int, int] = (96, 96),
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
        self._processor = ImageProcessor(image_type, resize, enable_norm=True)
        return self

    def set_custom_block(self, entry_point: str, kwargs: dict, processor: Optional[ObservationProcessor] = None):
        self._name = "custom"
        self._kwargs = dict(entry_point=entry_point, kwargs=kwargs)
        self._processor = processor
        return self

    # ----------------------------------------------------------------

    def get_processor(self) -> Optional[ObservationProcessor]:
        return self._processor

    def create_block_tf(self, enable_rnn: bool = False, flatten: bool = False):
        if self._name == "DQN":
            from srl.rl.models.tf.blocks.dqn_image_block import DQNImageBlock

            return DQNImageBlock(
                enable_rnn=enable_rnn,
                **self._kwargs,
            )
        if self._name == "R2D3":
            from srl.rl.models.tf.blocks.r2d3_image_block import R2D3ImageBlock

            return R2D3ImageBlock(
                enable_rnn=enable_rnn,
                **self._kwargs,
            )
        if self._name == "AlphaZero":
            from srl.rl.models.tf.blocks.alphazero_image_block import AlphaZeroImageBlock

            return AlphaZeroImageBlock(
                enable_rnn=enable_rnn,
                **self._kwargs,
            )
        if self._name == "MuzeroAtari":
            from srl.rl.models.tf.blocks.muzero_atari_block import MuZeroAtariBlock

            return MuZeroAtariBlock(
                enable_rnn=enable_rnn,
                **self._kwargs,
            )

        if self._name == "custom":
            from srl.utils.common import load_module

            return load_module(self._kwargs["entry_point"])(enable_rnn=enable_rnn, **self._kwargs["kwargs"])

        raise UndefinedError(self._name)

    def create_block_torch(self, in_shape: Tuple[int, ...], flatten: bool = False):
        if self._name == "DQN":
            from srl.rl.models.torch_.blocks.dqn_image_block import DQNImageBlock

            return DQNImageBlock(in_shape, flatten=flatten, **self._kwargs)
        if self._name == "R2D3":
            from srl.rl.models.torch_.blocks.r2d3_image_block import R2D3ImageBlock

            return R2D3ImageBlock(in_shape, flatten=flatten, **self._kwargs)

        if self._name == "AlphaZero":
            from srl.rl.models.torch_.blocks.alphazero_image_block import AlphaZeroImageBlock

            return AlphaZeroImageBlock(in_shape, **self._kwargs)
        if self._name == "MuzeroAtari":
            from srl.rl.models.torch_.blocks.muzero_atari_block import MuZeroAtariBlock

            return MuZeroAtariBlock(in_shape, **self._kwargs)

        if self._name == "custom":
            from srl.utils.common import load_module

            return load_module(self._kwargs["entry_point"])(**self._kwargs["kwargs"])

        raise UndefinedError(self._name)
