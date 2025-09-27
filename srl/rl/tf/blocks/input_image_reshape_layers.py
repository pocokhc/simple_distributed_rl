from tensorflow import keras

from srl.base.define import SpaceTypes
from srl.base.exception import TFLayerError, UndefinedError
from srl.base.spaces.box import BoxSpace

kl = keras.layers


def create_input_image_reshape_layers(space: BoxSpace, rnn: bool) -> list:
    err_msg = f"unknown space_type: {space}"
    layers = []

    if space.stype == SpaceTypes.GRAY_HW:
        if len(space.shape) == 2:
            # (h, w) -> (h, w, 1)
            layers.append(kl.Reshape(space.shape + (1,)))
        elif len(space.shape) == 3:
            # (len, h, w) -> (h, w, len)
            layers.append(kl.Permute((2, 3, 1)))
        else:
            raise TFLayerError(err_msg)

    elif space.stype == SpaceTypes.GRAY_HW1:
        assert space.shape[-1] == 1
        if len(space.shape) == 3:
            # (h, w, 1)
            pass
        elif len(space.shape) == 4:
            # (len, h, w, 1) -> (len, h, w)
            # (len, h, w) -> (h, w, len)
            layers.append(kl.Reshape(space.shape[:3]))
            layers.append(kl.Permute((2, 3, 1)))
        else:
            raise TFLayerError(err_msg)

    elif space.stype == SpaceTypes.RGB:
        if len(space.shape) == 3:
            # (h, w, ch)
            pass
        else:
            raise TFLayerError(err_msg)

    elif space.stype == SpaceTypes.FEATURE_MAP:
        # (h, w, ch)
        pass
    else:
        raise UndefinedError(space.stype)

    if rnn:
        layers = [kl.TimeDistributed(x) for x in layers]

    return layers
