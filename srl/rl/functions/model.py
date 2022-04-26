import enum
import logging
from typing import Tuple

from srl.base.define import EnvObservationType
from tensorflow.keras import layers as kl
from tensorflow.keras.regularizers import l2

logger = logging.getLogger(__name__)


class ImageLayerType(enum.Enum):
    NONE = 0
    DQN = enum.auto()
    R2D3 = enum.auto()
    AlphaZero = enum.auto()


def create_input_layers_one_sequence(
    input_shape: tuple,
    observation_type: EnvObservationType,
    image_layer_type: ImageLayerType,
) -> Tuple[kl.Layer, kl.Layer]:

    in_state = c = kl.Input(shape=input_shape)

    if observation_type == EnvObservationType.DISCRETE or observation_type == EnvObservationType.CONTINUOUS:
        c = kl.Flatten()(c)
        return in_state, c

    # --- image head
    if observation_type == EnvObservationType.GRAY_2ch:
        assert len(input_shape) == 2

        # (w, h) -> (w, h, 1)
        c = kl.Reshape(input_shape + (1,))(c)

    elif observation_type == EnvObservationType.GRAY_3ch:
        assert len(input_shape) == 3

        # (width, height, 1)
        pass

    elif observation_type == EnvObservationType.COLOR:
        assert len(input_shape) == 3

        # (width, height, ch)
        pass

    elif observation_type == EnvObservationType.SHAPE2:
        assert len(input_shape) == 2

        # (width, height) -> (width, height, 1)
        c = kl.Reshape(input_shape + (1,))(c)

    elif observation_type == EnvObservationType.SHAPE3:
        assert len(input_shape) == 3

        # (n, width, height) -> (width, height, n)
        c = kl.Permute((2, 3, 1))(c)

    else:
        raise ValueError()

    # --- image layers
    c = _create_image_layers(c, image_layer_type)

    return in_state, c


def create_input_layers(
    input_sequence: int,
    input_shape: tuple,
    observation_type: EnvObservationType,
    image_layer_type: ImageLayerType,
) -> Tuple[kl.Layer, kl.Layer]:

    input_shape = (input_sequence,) + input_shape
    in_state = c = kl.Input(shape=input_shape)

    if observation_type == EnvObservationType.DISCRETE or observation_type == EnvObservationType.CONTINUOUS:
        c = kl.Flatten()(c)
        return in_state, c

    # --- image head
    if input_sequence == 1:

        if observation_type == EnvObservationType.GRAY_2ch:
            assert len(input_shape) == 3

            # (1, w, h) -> (w, h, 1)
            c = kl.Permute((2, 3, 1))(c)

        elif observation_type == EnvObservationType.GRAY_3ch:
            assert len(input_shape) == 4

            # (1, width, height, 1) -> ( width, height, 1)
            c = kl.Reshape(input_shape[1:])(c)

        elif observation_type == EnvObservationType.COLOR:
            assert len(input_shape) == 4

            # (1, width, height, ch) -> (width, height, ch)
            c = kl.Reshape(input_shape[1:])(c)

        elif observation_type == EnvObservationType.SHAPE2:
            assert len(input_shape) == 3

            # (1, width, height) -> (width, height, 1)
            c = kl.Permute((2, 3, 1))(c)

        elif observation_type == EnvObservationType.SHAPE3:
            assert len(input_shape) == 4

            # (1, n, width, height) -> (n, width, height)
            # (n, width, height) -> (width, height, n)
            c = kl.Reshape(input_shape[1:])(c)
            c = kl.Permute((2, 3, 1))(c)

        else:
            raise ValueError()

    else:  # input_sequence > 1

        if observation_type == EnvObservationType.GRAY_2ch:
            assert len(input_shape) == 3

            # (in_stateseq, w, h) -> (w, h, in_stateseq)
            c = kl.Permute((2, 3, 1))(c)

        elif observation_type == EnvObservationType.GRAY_3ch:
            assert len(input_shape) == 4
            assert input_shape[-1] == 1

            # (in_stateseq, width, height, 1) -> (in_stateseq, width, height)
            # (in_stateseq, width, height) -> (width, height, in_stateseq)
            c = kl.Reshape(input_shape[:3])(c)
            c = kl.Permute((2, 3, 1))(c)

        elif observation_type == EnvObservationType.COLOR:
            raise ValueError()

        elif observation_type == EnvObservationType.SHAPE2:
            assert len(input_shape) == 3

            # (in_stateseq, width, height) -> (w, h, in_stateseq)
            c = kl.Permute((2, 3, 1))(c)

        elif observation_type == EnvObservationType.SHAPE3:
            raise ValueError()

        else:
            raise ValueError()

    # --- image layers
    c = _create_image_layers(c, image_layer_type)

    return in_state, c


def create_input_layers_lstm_stateful(
    batch_size: int,
    input_sequence: int,
    input_shape: tuple,
    observation_type: EnvObservationType,
    image_layer_type: ImageLayerType,
) -> Tuple[kl.Layer, kl.Layer]:

    input_shape = (input_sequence,) + input_shape
    in_state = c = kl.Input(batch_input_shape=(batch_size,) + input_shape)

    if observation_type == EnvObservationType.DISCRETE or observation_type == EnvObservationType.CONTINUOUS:
        c = kl.TimeDistributed(kl.Flatten())(c)
        return in_state, c

    if observation_type == EnvObservationType.GRAY_2ch:
        assert len(input_shape) == 3

        # (timesteps, w, h) -> (timesteps, w, h, 1)
        c = kl.Reshape(input_shape + (-1,))(c)

    elif observation_type == EnvObservationType.GRAY_3ch:
        assert len(input_shape) == 4
        assert input_shape[-1] == 1

        # (timesteps, width, height, 1)
        pass

    elif observation_type == EnvObservationType.COLOR:
        assert len(input_shape) == 4

        # (timesteps, width, height, ch)
        pass

    elif observation_type == EnvObservationType.SHAPE2:
        assert len(input_shape) == 3

        # (timesteps, width, height) -> (timesteps, width, height, 1)
        c = kl.Reshape(input_shape + (-1,))(c)

    elif observation_type == EnvObservationType.SHAPE3:
        assert len(input_shape) == 4

        # (timesteps, n, width, height) -> (timesteps, width, height, n)
        c = kl.Permute((1, 3, 4, 2))(c)

    else:
        raise ValueError()

    # --- image layers
    c = _create_image_layers(c, image_layer_type, use_lstm=True)

    return in_state, c


def _create_image_layers(c, image_layer_type, use_lstm: bool = False):

    # --- Image Layers
    if image_layer_type == ImageLayerType.NONE:
        pass

    elif image_layer_type == ImageLayerType.DQN:
        # --- DQN Image Model
        c = kl.Conv2D(32, (8, 8), strides=(4, 4), padding="same", activation="relu")(c)
        c = kl.Conv2D(64, (4, 4), strides=(2, 2), padding="same", activation="relu")(c)
        c = kl.Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="relu")(c)

    elif image_layer_type == ImageLayerType.R2D3:
        # --- R2D3 Image Model
        # https://arxiv.org/abs/1909.01387
        c = _resblock(c, 16)
        c = _resblock(c, 32)
        c = _resblock(c, 32)
        c = kl.Activation("relu")(c)

        # TODO: lstm

    elif image_layer_type == ImageLayerType.AlphaZero:
        c = _resblock(c, 16)
        c = _resblock(c, 32)
        c = _resblock(c, 32)
        c = kl.Activation("relu")(c)

        # TODO

    else:
        raise ValueError()

    if use_lstm:
        c = kl.TimeDistributed(kl.Flatten())(c)
    else:
        c = kl.Flatten()(c)

    return c


def _resblock(c, n_filter):
    c = kl.Conv2D(n_filter, (3, 3), strides=(1, 1), padding="same")(c)
    c = kl.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(c)
    c = _residual_block(c, n_filter)
    c = _residual_block(c, n_filter)
    return c


def _residual_block(c, n_filter):

    c_tmp = kl.Activation("relu")(c)
    c_tmp = kl.Conv2D(n_filter, (3, 3), strides=(1, 1), padding="same")(c_tmp)

    c_tmp = kl.Activation("relu")(c_tmp)
    c_tmp = kl.Conv2D(n_filter, (3, 3), strides=(1, 1), padding="same")(c)

    # 結合
    c = kl.Add()([c, c_tmp])
    return c


def _resblock2(c, filters):
    c_tmp = kl.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        use_bias=True,
        kernel_regularizer=l2(0.001),
        kernel_initializer="he_normal",
    )(c)
    c_tmp = kl.BatchNormalization()(c_tmp)
    c_tmp = kl.Activation("relu")(c_tmp)
    c_tmp = kl.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        use_bias=True,
        kernel_regularizer=l2(0.001),
        kernel_initializer="he_normal",
    )(c_tmp)
    c_tmp = kl.BatchNormalization()(c_tmp)
    c = kl.Add()([c, c_tmp])
    c = kl.Activation("relu")(c)
    return c
