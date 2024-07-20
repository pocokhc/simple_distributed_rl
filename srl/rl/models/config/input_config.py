from dataclasses import dataclass, field
from typing import List, Optional, Tuple, cast

from srl.base.define import SpaceTypes
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.space import SpaceBase
from srl.rl.models.config.mlp_block import MLPBlockConfig
from srl.rl.processors.image_processor import ImageProcessor


@dataclass
class RLConfigComponentInput:

    #: <:ref:`MLPBlockConfig`>
    input_value_block: "MLPBlockConfig" = field(init=False, default_factory=lambda: MLPBlockConfig().set(()))

    #: <:ref:`InputImageBlockConfig`>
    input_image_block: "InputImageBlockConfig" = field(init=False, default_factory=lambda: InputImageBlockConfig())

    def get_processors(self) -> List[RLProcessor]:
        return self.input_image_block.get_processors()

    def assert_params_input(self):
        pass

    # ----------------------------------------------------------------

    def create_input_block_tf(
        self,
        image_flatten: bool = True,
        rnn: bool = False,
        out_multi: bool = False,
        in_space: Optional[SpaceBase] = None,
        **kwargs,
    ):
        from srl.rl.tf.blocks.input_block import create_block_from_config

        if in_space is None:
            assert hasattr(self, "observation_space")
            in_space = cast(SpaceBase, cast(RLConfig, self).observation_space)

        return create_block_from_config(self, in_space, image_flatten, rnn, out_multi, **kwargs)

    def create_input_block_torch(
        self,
        image_flatten: bool = True,
        out_multi: bool = False,
        in_space: Optional[SpaceBase] = None,
        **kwargs,
    ):
        from srl.rl.torch_.blocks.input_block import create_block_from_config

        if in_space is None:
            assert hasattr(self, "observation_space")
            in_space = cast(SpaceBase, cast(RLConfig, self).observation_space)

        return create_block_from_config(self, in_space, image_flatten, out_multi, **kwargs)


@dataclass
class InputImageBlockConfig:
    _name: str = ""
    _kwargs: dict = field(default_factory=dict)
    _processors: List[RLProcessor] = field(default_factory=list)

    def __post_init__(self):
        if self._name == "":
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
        self._processors = [ImageProcessor(image_type, resize, enable_norm=True)]
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
        self._processors = [ImageProcessor(image_type, resize, enable_norm=True)]
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
        self._processors = []
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
        self._processors = [ImageProcessor(image_type, resize, enable_norm=True)]
        return self

    def set_custom_block(self, entry_point: str, kwargs: dict, processors: List[RLProcessor] = []):
        self._name = "custom"
        self._kwargs = dict(entry_point=entry_point, kwargs=kwargs)
        self._processors = processors[:]
        return self

    # ----------------------------------------------------------------

    def get_processors(self) -> List[RLProcessor]:
        return self._processors
