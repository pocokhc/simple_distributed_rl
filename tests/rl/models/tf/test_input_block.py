import numpy as np
import pytest

from srl.base.define import EnvObservationTypes
from srl.base.exception import TFLayerError


def call_block_tf(
    obs_shape,
    obs_type,
    x,
    enable_time_distributed_layer=False,
):
    from srl.rl.models.tf.input_block import InputImageBlock

    block = InputImageBlock(
        obs_shape,
        obs_type,
        enable_time_distributed_layer=enable_time_distributed_layer,
    )
    y = block(x)
    assert y is not None
    y = y.numpy()
    return y, block.use_image_layer, None


pattern0 = [
    ((2, 4, 8), EnvObservationTypes.UNKNOWN, (2 * 4 * 8,), False),
    ((2, 4, 8), EnvObservationTypes.DISCRETE, (2 * 4 * 8,), False),
    ((2, 4, 8), EnvObservationTypes.CONTINUOUS, (2 * 4 * 8,), False),
    ((4, 8), EnvObservationTypes.GRAY_2ch, (4, 8, 1), True),
    ((4, 8, 1), EnvObservationTypes.GRAY_3ch, (4, 8, 1), True),
    ((4, 8, 3), EnvObservationTypes.COLOR, (4, 8, 3), True),
]


@pytest.mark.parametrize("obs_shape, obs_type, true_shape, true_image", pattern0)
def test_window_0_tf(obs_shape, obs_type, true_shape, true_image):
    pytest.importorskip("tensorflow")
    _window_0(call_block_tf, obs_shape, obs_type, true_shape, true_image)


def _window_0(call_block, obs_shape, obs_type, true_shape, true_image):
    batch_size = 8

    x = np.ones((batch_size,) + obs_shape, dtype=np.float32)
    y, use_image_layer, out_shape = call_block(obs_shape, obs_type, x)
    assert use_image_layer == true_image
    assert y.shape == (batch_size,) + true_shape
    if out_shape is not None:
        assert out_shape == true_shape


pattern10 = [
    ((10, 2, 4, 8), EnvObservationTypes.UNKNOWN, (10 * 2 * 4 * 8,), False, False),
    ((10, 2, 4, 8), EnvObservationTypes.DISCRETE, (10 * 2 * 4 * 8,), False, False),
    ((10, 2, 4, 8), EnvObservationTypes.CONTINUOUS, (10 * 2 * 4 * 8,), False, False),
    ((10, 4, 8), EnvObservationTypes.GRAY_2ch, (4, 8, 10), True, False),
    ((10, 4, 8, 1), EnvObservationTypes.GRAY_3ch, (4, 8, 10), True, False),
    ((10, 4, 8, 3), EnvObservationTypes.COLOR, None, None, True),
]


@pytest.mark.parametrize("obs_shape, obs_type, true_shape, true_image, is_throw", pattern10)
def test_window_10_tf(obs_shape, obs_type, true_shape, true_image, is_throw):
    pytest.importorskip("tensorflow")
    _window_10(call_block_tf, obs_shape, obs_type, true_shape, true_image, is_throw)


def _window_10(call_block, obs_shape, obs_type, true_shape, true_image, is_throw):
    batch_size = 8

    x = np.ones((batch_size,) + obs_shape, dtype=np.float32)
    if is_throw:
        with pytest.raises(TFLayerError):
            call_block(obs_shape, obs_type, x)
    else:
        y, use_image_layer, out_shape = call_block(obs_shape, obs_type, x)
        assert use_image_layer == true_image
        assert y.shape == (batch_size,) + true_shape
        if out_shape is not None:
            assert out_shape == true_shape


pattern_time = [
    ((2, 4, 8), EnvObservationTypes.UNKNOWN, (2 * 4 * 8,), False),
    ((2, 4, 8), EnvObservationTypes.DISCRETE, (2 * 4 * 8,), False),
    ((2, 4, 8), EnvObservationTypes.CONTINUOUS, (2 * 4 * 8,), False),
    ((4, 8), EnvObservationTypes.GRAY_2ch, (4, 8, 1), True),
    ((4, 8, 1), EnvObservationTypes.GRAY_3ch, (4, 8, 1), True),
    ((4, 8, 3), EnvObservationTypes.COLOR, (4, 8, 3), True),
]


@pytest.mark.parametrize("obs_shape, obs_type, true_shape, true_image", pattern_time)
def test_time_layer_tf(obs_shape, obs_type, true_shape, true_image):
    pytest.importorskip("tensorflow")
    _time_layer(call_block_tf, obs_shape, obs_type, true_shape, true_image)


def _time_layer(call_block, obs_shape, obs_type, true_shape, true_image):
    batch_size = 7
    seq_len = 3

    x = np.ones((batch_size, seq_len) + obs_shape, dtype=np.float32)
    y, use_image_layer, out_shape = call_block(obs_shape, obs_type, x, enable_time_distributed_layer=True)
    assert use_image_layer == true_image
    assert y.shape == (batch_size, seq_len) + true_shape
    if out_shape is not None:
        assert out_shape == true_shape
