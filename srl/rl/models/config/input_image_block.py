from dataclasses import dataclass, field
from typing import List, Tuple

from srl.base.define import SpaceTypes
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.space import SpaceBase
from srl.rl.processors.image_processor import ImageProcessor


@dataclass
class InputImageBlockConfig:
    name: str = ""
    kwargs: dict = field(default_factory=dict)
    processors: List[RLProcessor] = field(default_factory=list)

    def __post_init__(self):
        if self.name == "":
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
        self.name = "DQN"
        self.kwargs = dict(filters=filters, activation=activation)
        self.processors = [ImageProcessor(image_type, resize, normalize_type="0to1")]
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
        self.name = "R2D3"
        self.kwargs = dict(filters=filters, activation=activation)
        self.processors = [ImageProcessor(image_type, resize, normalize_type="0to1")]
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
        self.name = "AlphaZero"
        self.kwargs = dict(
            n_blocks=n_blocks,
            filters=filters,
            activation=activation,
        )
        self.processors = []
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
        self.name = "MuzeroAtari"
        self.kwargs = dict(
            filters=filters,
            activation=activation,
            use_layer_normalization=use_layer_normalization,
        )
        self.processors = [ImageProcessor(image_type, resize, normalize_type="0to1")]
        return self

    def set_custom_block(self, entry_point: str, kwargs: dict, processors: List[RLProcessor] = []):
        self.name = "custom"
        self.kwargs = dict(entry_point=entry_point, kwargs=kwargs)
        self.processors = processors[:]
        return self

    # ----------------------------------------------------------------

    def get_processors(self) -> List[RLProcessor]:
        return self.processors

    def create_tf_block(self, in_space: SpaceBase, out_flatten: bool = True, rnn: bool = False, **kwargs):
        from srl.rl.tf.blocks.input_image_block import InputImageBlock

        return InputImageBlock(self, in_space, out_flatten, rnn, **kwargs)

    def create_torch_block(self, in_space: SpaceBase, out_flatten: bool = True, reshape_for_rnn: bool = False):
        from srl.rl.torch_.blocks.input_image_block import InputImageBlock

        return InputImageBlock(self, in_space, out_flatten, reshape_for_rnn)
