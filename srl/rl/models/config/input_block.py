from dataclasses import dataclass, field
from typing import List, Optional, Tuple, cast

import numpy as np

from srl.base.define import SpaceTypes
from srl.base.exception import UndefinedError
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.space import SpaceBase
from srl.rl.processors.image_processor import ImageProcessor


@dataclass
class InputValueBlockConfig:
    name: str = ""
    kwargs: dict = field(default_factory=dict)
    processors: List[RLProcessor] = field(default_factory=list)

    def __post_init__(self):
        self.set()

    def set(self, layer_sizes: Tuple[int, ...] = (), activation: str = "relu"):
        self.name = "MLP"
        self.kwargs = {
            "layer_sizes": layer_sizes,
            "activation": activation,
        }
        self.processors = []
        return self

    def set_custom_block(self, entry_point: str, kwargs: dict, processors: List[RLProcessor] = []):
        self.name = "custom"
        self.kwargs = {
            "entry_point": entry_point,
            "kwargs": kwargs,
        }
        self.processors = processors
        return self

    # ----------------------------------------------------------------
    def get_processors(self) -> List[RLProcessor]:
        return self.processors

    def create_tf_block(self, rnn: bool = False, **kwargs):
        from tensorflow.keras import layers

        from srl.rl.tf.model import SequentialModel

        kwargs2 = self.kwargs.copy()
        kwargs2.update(kwargs)
        blocks = []

        # --- input flatten
        if rnn:
            blocks.append(layers.TimeDistributed(layers.Flatten()))
        else:
            blocks.append(layers.Flatten())

        # --- MLP
        if self.name == "MLP":
            from srl.rl.tf.blocks.mlp_block import MLPBlock

            blocks.append(MLPBlock(**kwargs2))
        elif self.name == "custom":
            from srl.utils.common import load_module

            kwargs2 = self.kwargs["kwargs"].copy()
            kwargs2.update(kwargs)
            blocks.append(load_module(self.kwargs["entry_point"])(rnn=rnn, **kwargs2))
        else:
            raise UndefinedError(self)

        return SequentialModel(blocks)

    def create_tf_dummy_data(self, cfg: RLConfig, batch_size: int = 1, timesteps: int = 0) -> np.ndarray:
        in_shape = cfg.observation_space.shape
        np_dtype = cfg.get_dtype("np")
        if timesteps > 0:
            return np.zeros((batch_size, timesteps) + in_shape, np_dtype)
        else:
            return np.zeros((batch_size,) + in_shape, np_dtype)

    def create_torch_block(self, in_space: BoxSpace, reshape_for_rnn: bool = False):
        if self.name == "MLP":
            from srl.rl.torch_.blocks.input_value_block import InputValueBlock

            return InputValueBlock(in_space.shape, input_flatten=True, reshape_for_rnn=reshape_for_rnn, **self.kwargs)

        if self.name == "custom":
            from srl.utils.common import load_module

            return load_module(self.kwargs["entry_point"])(in_space.shape, input_flatten=True, reshape_for_rnn=reshape_for_rnn, **self.kwargs["kwargs"])

        raise UndefinedError(self)


@dataclass
class InputImageBlockConfig:
    name: str = ""
    kwargs: dict = field(default_factory=dict)
    processors: Optional[List[RLProcessor]] = None

    def __post_init__(self):
        self.set_dqn_block()

    def set_dqn_block(self, filters: int = 32, activation: str = "relu"):
        """画像の入力に対してDQNで採用されたLayersを使用します。

        Args:
            filters (int, optional): 基準となるfilterの数です.
            activation (str): activation function. Defaults to "relu".
        """
        self.name = "DQN"
        self.processors = None
        self.kwargs = {
            "filters": filters,
            "activation": activation,
        }
        return self

    def set_r2d3_block(self, filters: int = 16, activation: str = "relu"):
        """画像の入力に対してR2D3で採用されたLayersを使用します。

        Args:
            filters (int, optional): 基準となるfilterの数です.
            activation (str, optional): activation function. Defaults to "relu".
        """
        self.name = "R2D3"
        self.processors = None
        self.kwargs = {
            "filters": filters,
            "activation": activation,
        }
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
        self.processors = []
        self.kwargs = {
            "n_blocks": n_blocks,
            "filters": filters,
            "activation": activation,
        }
        return self

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
        self.name = "MuzeroAtari"
        self.processors = None
        self.kwargs = {
            "filters": filters,
            "activation": activation,
            "use_layer_normalization": use_layer_normalization,
        }
        return self

    def set_custom_block(self, entry_point: str, kwargs: dict, processors: List[RLProcessor] = []):
        self.name = "custom"
        self.processors = processors
        self.kwargs = {
            "entry_point": entry_point,
            "kwargs": kwargs,
        }
        return self

    # ----------------------------------------------------------------

    def get_processors(self) -> List[RLProcessor]:
        if self.processors is not None:
            return self.processors

        if self.name == "DQN":
            return [ImageProcessor(SpaceTypes.GRAY_HW1, (84, 84), normalize_type="0to1")]
        elif self.name == "R2D3":
            return [ImageProcessor(SpaceTypes.RGB, (96, 72), normalize_type="0to1")]
        elif self.name == "MuzeroAtari":
            return [ImageProcessor(SpaceTypes.RGB, (96, 96), normalize_type="0to1")]
        else:
            raise UndefinedError(self)

    def create_tf_block(self, in_space: BoxSpace, out_flatten: bool = True, rnn: bool = False, **kwargs):
        from tensorflow.keras import layers

        kwargs2 = self.kwargs.copy()
        kwargs2.update(kwargs)

        # --- reshape
        from srl.rl.tf.blocks.input_image_reshape_layers import create_input_image_reshape_layers
        from srl.rl.tf.model import SequentialModel

        blocks = create_input_image_reshape_layers(in_space, rnn)

        # --- encode image
        if self.name == "DQN":
            from srl.rl.tf.blocks.dqn_image_block import DQNImageBlock

            blocks.append(DQNImageBlock(rnn=rnn, **kwargs2))
        elif self.name == "R2D3":
            from srl.rl.tf.blocks.r2d3_image_block import R2D3ImageBlock

            blocks.append(R2D3ImageBlock(rnn=rnn, **kwargs2))
        elif self.name == "AlphaZero":
            from srl.rl.tf.blocks.alphazero_image_block import AlphaZeroImageBlock

            blocks.append(AlphaZeroImageBlock(**kwargs2))
        elif self.name == "MuzeroAtari":
            from srl.rl.tf.blocks.muzero_atari_block import MuZeroAtariBlock

            blocks.append(MuZeroAtariBlock(**kwargs2))
        elif self.name == "custom":
            from srl.utils.common import load_module

            kwargs2 = self.kwargs["kwargs"].copy()
            kwargs2.update(kwargs)
            blocks.append(load_module(self.kwargs["entry_point"])(rnn=rnn, **kwargs2))
        else:
            raise UndefinedError(self)

        # --- out flatten
        if out_flatten:
            if rnn:
                blocks.append(layers.TimeDistributed(layers.Flatten()))
            else:
                blocks.append(layers.Flatten())

        return SequentialModel(blocks)

    def create_tf_dummy_data(self, cfg: RLConfig, batch_size: int = 1, timesteps: int = 0) -> np.ndarray:
        in_shape = cfg.observation_space.shape
        np_dtype = cfg.get_dtype("np")
        if timesteps > 0:
            return np.zeros((batch_size, timesteps) + in_shape, np_dtype)
        else:
            return np.zeros((batch_size,) + in_shape, np_dtype)

    def create_torch_block(self, in_space: BoxSpace, out_flatten: bool = True, reshape_for_rnn: bool = False):
        from srl.rl.torch_.blocks.input_image_block import InputImageBlock

        return InputImageBlock(self, in_space, out_flatten, reshape_for_rnn)


@dataclass
class InputBlockConfig:
    value: InputValueBlockConfig = field(default_factory=lambda: InputValueBlockConfig())
    image: InputImageBlockConfig = field(default_factory=lambda: InputImageBlockConfig())

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        if prev_observation_space.is_value():
            return self.value.get_processors()
        elif prev_observation_space.is_image_like():
            return self.image.get_processors()
        return []

    def create_tf_block(self, cfg: RLConfig, out_flatten: bool = True, rnn: bool = False, **kwargs):
        in_space = cast(BoxSpace, cfg.observation_space)
        if in_space.is_value():
            return self.value.create_tf_block(rnn, **kwargs)
        elif in_space.is_image_like():
            return self.image.create_tf_block(in_space, out_flatten, rnn, **kwargs)
        else:
            raise UndefinedError(in_space)

    def create_tf_dummy_data(self, cfg: RLConfig, batch_size: int = 1, timesteps: int = 0) -> np.ndarray:
        in_shape = cfg.observation_space.shape
        np_dtype = cfg.get_dtype("np")
        if timesteps > 0:
            return np.zeros((batch_size, timesteps) + in_shape, np_dtype)
        else:
            return np.zeros((batch_size,) + in_shape, np_dtype)

    def create_torch_block(self, cfg: RLConfig, out_flatten: bool = True, reshape_for_rnn: bool = False):
        in_space = cast(BoxSpace, cfg.observation_space)
        if in_space.is_value():
            return self.value.create_torch_block(in_space, reshape_for_rnn)
        elif in_space.is_image_like():
            return self.image.create_torch_block(in_space, out_flatten, reshape_for_rnn)
        else:
            raise UndefinedError(in_space)
