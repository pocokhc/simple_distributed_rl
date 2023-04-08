import logging
import warnings
from typing import List, Tuple

import tensorflow.keras as keras
from tensorflow.keras import layers as kl

from srl.base.define import EnvObservationType

logger = logging.getLogger(__name__)


class InputBlock(keras.Model):
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        observation_type: EnvObservationType,
        enable_time_distributed_layer: bool = False,
    ):
        super().__init__()
        self._init_layer(observation_shape, observation_type)

        if enable_time_distributed_layer:
            self.in_layers = [kl.TimeDistributed(x) for x in self.in_layers]

    def _init_layer(self, observation_shape, observation_type):
        err_msg = f"unknown observation_type: {observation_type}"
        self.in_layers = []
        self.use_image_layer = False

        # --- value head
        if (
            observation_type == EnvObservationType.DISCRETE
            or observation_type == EnvObservationType.CONTINUOUS
            or observation_type == EnvObservationType.UNKNOWN
        ):
            self.in_layers.append(kl.Flatten())
            return

        self.use_image_layer = True

        # --- image head
        if observation_type == EnvObservationType.GRAY_2ch:
            if len(observation_shape) == 2:
                # (w, h) -> (w, h, 1)
                self.in_layers.append(kl.Reshape(observation_shape + (1,)))
            elif len(observation_shape) == 3:
                # (len, w, h) -> (w, h, len)
                self.in_layers.append(kl.Permute((2, 3, 1)))
            else:
                raise ValueError(err_msg)

        elif observation_type == EnvObservationType.GRAY_3ch:
            assert observation_shape[-1] == 1
            if len(observation_shape) == 3:
                # (w, h, 1)
                pass
            elif len(observation_shape) == 4:
                # (len, w, h, 1) -> (len, w, h)
                # (len, w, h) -> (w, h, len)
                self.in_layers.append(kl.Reshape(observation_shape[:3]))
                self.in_layers.append(kl.Permute((2, 3, 1)))
            else:
                raise ValueError(err_msg)

        elif observation_type == EnvObservationType.COLOR:
            if len(observation_shape) == 3:
                # (w, h, ch)
                pass
            else:
                raise ValueError(err_msg)

        elif observation_type == EnvObservationType.SHAPE2:
            if len(observation_shape) == 2:
                # (w, h) -> (w, h, 1)
                self.in_layers.append(kl.Reshape(observation_shape + (1,)))
            elif len(observation_shape) == 3:
                # (len, w, h) -> (w, h, len)
                self.in_layers.append(kl.Permute((2, 3, 1)))
            else:
                raise ValueError(err_msg)

        elif observation_type == EnvObservationType.SHAPE3:
            if len(observation_shape) == 3:
                # (n, w, h) -> (w, h, n)
                self.in_layers.append(kl.Permute((2, 3, 1)))
            else:
                raise ValueError(err_msg)

        else:
            raise ValueError(err_msg)

    def call(self, x, training=False):
        for layer in self.in_layers:
            x = layer(x, training=training)
        return x

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(self.__input_shape)

    def init_model_graph(self, name: str = ""):
        x = kl.Input(shape=self.__input_shape[1:])
        name = self.__class__.__name__ if name == "" else name
        keras.Model(inputs=x, outputs=self.call(x), name=name)


def create_input_layers(
    observation_shape: Tuple[int, ...],
    observation_type: EnvObservationType,
) -> Tuple[List[kl.Layer], bool]:
    """状態の入力レイヤーを作成してそのリストを返す

    Args:
        observation_shape (Tuple[int, ...]): 状態の入力shape
        observation_type (EnvObservationType): 状態が何かを表すEnvObservationType

    Returns:
        [
            in_layers  (kl.Layer): 入力に使うlayerのlistを返します
            use_image_head (bool):
                Falseの時 out_layer は flatten、
                Trueの時 out_layer は CNN の形式で返ります。
        ]
    """
    # warnings.warn(
    #    "'FunctionalAPI' was changed to 'SubclassingAPI'. This function will be removed in the next update.",
    #    DeprecationWarning,
    # )
    err_msg = f"unknown observation_type: {observation_type}"

    layers = []

    # --- value head
    if (
        observation_type == EnvObservationType.DISCRETE
        or observation_type == EnvObservationType.CONTINUOUS
        or observation_type == EnvObservationType.UNKNOWN
    ):
        layers.append(kl.Flatten())
        return layers, False

    # --- image head
    if observation_type == EnvObservationType.GRAY_2ch:
        if len(observation_shape) == 2:
            # (w, h) -> (w, h, 1)
            layers.append(kl.Reshape(observation_shape + (1,)))
        elif len(observation_shape) == 3:
            # (len, w, h) -> (w, h, len)
            layers.append(kl.Permute((2, 3, 1)))
        else:
            raise ValueError(err_msg)

    elif observation_type == EnvObservationType.GRAY_3ch:
        assert observation_shape[-1] == 1
        if len(observation_shape) == 3:
            # (w, h, 1)
            pass
        elif len(observation_shape) == 4:
            # (len, w, h, 1) -> (len, w, h)
            # (len, w, h) -> (w, h, len)
            layers.append(kl.Reshape(observation_shape[:3]))
            layers.append(kl.Permute((2, 3, 1)))
        else:
            raise ValueError(err_msg)

    elif observation_type == EnvObservationType.COLOR:
        if len(observation_shape) == 3:
            # (w, h, ch)
            pass
        else:
            raise ValueError(err_msg)

    elif observation_type == EnvObservationType.SHAPE2:
        if len(observation_shape) == 2:
            # (w, h) -> (w, h, 1)
            layers.append(kl.Reshape(observation_shape + (1,)))
        elif len(observation_shape) == 3:
            # (len, w, h) -> (w, h, len)
            layers.append(kl.Permute((2, 3, 1)))
        else:
            raise ValueError(err_msg)

    elif observation_type == EnvObservationType.SHAPE3:
        if len(observation_shape) == 3:
            # (n, w, h) -> (w, h, n)
            layers.append(kl.Permute((2, 3, 1)))
        else:
            raise ValueError(err_msg)

    else:
        raise ValueError(err_msg)

    return layers, True


def create_input_layer(
    observation_shape: Tuple[int, ...],
    observation_type: EnvObservationType,
) -> Tuple[kl.Layer, kl.Layer, bool]:
    """状態の入力レイヤーを作成して返します

    Args:
        observation_shape (Tuple[int, ...]): 状態の入力shape
        observation_type (EnvObservationType): 状態が何かを表すEnvObservationType

    Returns:
        [
            in_layer  (kl.Layer): modelの入力に使うlayerを返します
            out_layer (kl.Layer): modelの続きに使うlayerを返します
            use_image_head (bool):
                Falseの時 out_layer は flatten、
                Trueの時 out_layer は CNN の形式で返ります。
        ]
    """
    warnings.warn(
        "'FunctionalAPI' was changed to 'SubclassingAPI'. This function will be removed in the next update.",
        DeprecationWarning,
    )

    # --- input
    in_layer = c = kl.Input(shape=observation_shape)
    err_msg = f"unknown observation_type: {observation_type}"

    # --- value head
    if (
        observation_type == EnvObservationType.DISCRETE
        or observation_type == EnvObservationType.CONTINUOUS
        or observation_type == EnvObservationType.UNKNOWN
    ):
        c = kl.Flatten()(c)
        return in_layer, c, False

    # --- image head
    if observation_type == EnvObservationType.GRAY_2ch:
        if len(observation_shape) == 2:
            # (w, h) -> (w, h, 1)
            c = kl.Reshape(observation_shape + (1,))(c)
        elif len(observation_shape) == 3:
            # (len, w, h) -> (w, h, len)
            c = kl.Permute((2, 3, 1))(c)
        else:
            raise ValueError(err_msg)

    elif observation_type == EnvObservationType.GRAY_3ch:
        assert observation_shape[-1] == 1
        if len(observation_shape) == 3:
            # (w, h, 1)
            pass
        elif len(observation_shape) == 4:
            # (len, w, h, 1) -> (len, w, h)
            # (len, w, h) -> (w, h, len)
            c = kl.Reshape(observation_shape[:3])(c)
            c = kl.Permute((2, 3, 1))(c)
        else:
            raise ValueError(err_msg)

    elif observation_type == EnvObservationType.COLOR:
        if len(observation_shape) == 3:
            # (w, h, ch)
            pass
        else:
            raise ValueError(err_msg)

    elif observation_type == EnvObservationType.SHAPE2:
        if len(observation_shape) == 2:
            # (w, h) -> (w, h, 1)
            c = kl.Reshape(observation_shape + (1,))(c)
        elif len(observation_shape) == 3:
            # (len, w, h) -> (w, h, len)
            c = kl.Permute((2, 3, 1))(c)
        else:
            raise ValueError(err_msg)

    elif observation_type == EnvObservationType.SHAPE3:
        if len(observation_shape) == 3:
            # (n, w, h) -> (w, h, n)
            c = kl.Permute((2, 3, 1))(c)
        else:
            raise ValueError(err_msg)

    else:
        raise ValueError(err_msg)

    return in_layer, c, True


def create_input_layer_stateful_lstm(
    batch_size: int,
    observation_shape: Tuple[int, ...],
    observation_type: EnvObservationType,
) -> Tuple[kl.Layer, kl.Layer, bool]:
    """状態の入力レイヤーを作成して返します。
    input_sequence は1で固定します。

    Args:
        batch_size (int): batch_size
        observation_shape (Tuple[int, ...]): 状態の入力shape
        observation_type (EnvObservationType): 状態が何かを表すEnvObservationType

    Returns:
        [
            in_layer  (kl.Layer): modelの入力に使うlayerを返します
            out_layer (kl.Layer): modelの続きに使うlayerを返します
            use_image_head (bool):
                Falseの時 out_layer は flatten、
                Trueの時 out_layer は CNN の形式で返ります。
        ]
    """
    warnings.warn(
        "'FunctionalAPI' was changed to 'SubclassingAPI'. This function will be removed in the next update.",
        DeprecationWarning,
    )

    # --- input
    input_shape = (1,) + observation_shape
    in_layer = c = kl.Input(batch_input_shape=(batch_size,) + input_shape)

    # --- value head
    if (
        observation_type == EnvObservationType.DISCRETE
        or observation_type == EnvObservationType.CONTINUOUS
        or observation_type == EnvObservationType.UNKNOWN
    ):
        c = kl.TimeDistributed(kl.Flatten())(c)
        return in_layer, c, False

    # --- image head
    if observation_type == EnvObservationType.GRAY_2ch:
        assert len(input_shape) == 3

        # (timesteps, w, h) -> (timesteps, w, h, 1)
        c = kl.Reshape(input_shape + (-1,))(c)

    elif observation_type == EnvObservationType.GRAY_3ch:
        assert len(input_shape) == 4
        assert input_shape[-1] == 1

        # (timesteps, width, height, 1)

    elif observation_type == EnvObservationType.COLOR:
        assert len(input_shape) == 4

        # (timesteps, width, height, ch)

    elif observation_type == EnvObservationType.SHAPE2:
        assert len(input_shape) == 3

        # (timesteps, width, height) -> (timesteps, width, height, 1)
        c = kl.Reshape(input_shape + (-1,))(c)

    elif observation_type == EnvObservationType.SHAPE3:
        assert len(input_shape) == 4

        # (timesteps, n, width, height) -> (timesteps, width, height, n)
        c = kl.Permute((1, 3, 4, 2))(c)

    else:
        raise ValueError(f"unknown observation_type: {observation_type}")

    return in_layer, c, True
