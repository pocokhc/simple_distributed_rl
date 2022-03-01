import enum

from srl.base.define import EnvObservationType
from tensorflow.keras import layers as kl


class ImageLayerType(enum.Enum):
    NONE = 0
    DQN = 1
    R2D3 = 2


def create_input_layers(
    input_sequence,
    input_shape,
    observation_type: EnvObservationType,
    image_layer_type: ImageLayerType,
) -> tuple[kl.Layer, kl.Layer]:

    input_shape = input_shape
    # input_shape = (input_sequence,) + input_shape

    if observation_type == EnvObservationType.DISCRETE or observation_type == EnvObservationType.CONTINUOUS:
        input_ = c = kl.Input(shape=input_shape)
        c = kl.Flatten()(c)
        return input_, c

    # --- image
    input_ = c = kl.Input(shape=input_shape)

    if observation_type == EnvObservationType.GRAY_2ch:
        assert len(input_shape) == 3
        # (input_seq, w, h) -> (w, h, input_seq)
        c = kl.Permute((2, 3, 1))(c)

    elif observation_type == EnvObservationType.SHAPE2:
        assert len(input_shape) == 3
        # (n, width, height) -> (w, h, n)
        c = kl.Permute((2, 3, 1))(c)
    else:
        # TODO
        # (input_seq, w, h, ch)
        # assert len(input_shape) == 4
        c = kl.Permute((2, 3, 1))(c)
        pass

    # --- Image Layers
    if image_layer_type == ImageLayerType.NONE:
        c = kl.Flatten()(c)

    elif image_layer_type == ImageLayerType.DQN:
        # --- DQN Image Model
        c = kl.Conv2D(32, (8, 8), strides=(4, 4), padding="same", activation="relu")(c)
        c = kl.Conv2D(64, (4, 4), strides=(2, 2), padding="same", activation="relu")(c)
        c = kl.Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="relu")(c)
        c = kl.Flatten()(c)

    elif image_layer_type == ImageLayerType.DQN:
        # --- R2D3 Image Model
        # https://arxiv.org/abs/1909.01387
        c = _resblock(c, 16)
        c = _resblock(c, 32)
        c = _resblock(c, 32)
        c = kl.Activation("relu")(c)
        c = kl.Flatten()(c)

    else:
        raise ValueError()

    return input_, c


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
