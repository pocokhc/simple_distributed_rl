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
    return y.numpy()


pattern0 = [
    # ((2, 4, 8), EnvObservationTypes.UNKNOWN, (2 * 4 * 8,)),
    # ((2, 4, 8), EnvObservationTypes.DISCRETE, (2 * 4 * 8,)),
    # ((2, 4, 8), EnvObservationTypes.CONTINUOUS, (2 * 4 * 8,)),
    ((4, 8), EnvObservationTypes.GRAY_2ch, (4, 8, 1)),
    ((4, 8, 1), EnvObservationTypes.GRAY_3ch, (4, 8, 1)),
    ((4, 8, 3), EnvObservationTypes.COLOR, (4, 8, 3)),
]


@pytest.mark.parametrize("obs_shape, obs_type, true_shape", pattern0)
def test_window_0_tf(obs_shape, obs_type, true_shape):
    pytest.importorskip("tensorflow")
    _window_0(call_block_tf, obs_shape, obs_type, true_shape)


def _window_0(call_block, obs_shape, obs_type, true_shape):
    batch_size = 8

    x = np.ones((batch_size,) + obs_shape, dtype=np.float32)
    y = call_block(obs_shape, obs_type, x)
    assert y.shape == (batch_size,) + true_shape


pattern10 = [
    # ((10, 2, 4, 8), EnvObservationTypes.UNKNOWN, (10 * 2 * 4 * 8,), False),
    # ((10, 2, 4, 8), EnvObservationTypes.DISCRETE, (10 * 2 * 4 * 8,), False),
    # ((10, 2, 4, 8), EnvObservationTypes.CONTINUOUS, (10 * 2 * 4 * 8,), False),
    ((10, 4, 8), EnvObservationTypes.GRAY_2ch, (4, 8, 10), False),
    ((10, 4, 8, 1), EnvObservationTypes.GRAY_3ch, (4, 8, 10), False),
    ((10, 4, 8, 3), EnvObservationTypes.COLOR, None, True),
]


@pytest.mark.parametrize("obs_shape, obs_type, true_shape, is_throw", pattern10)
def test_window_10_tf(obs_shape, obs_type, true_shape, is_throw):
    pytest.importorskip("tensorflow")
    _window_10(call_block_tf, obs_shape, obs_type, true_shape, is_throw)


def _window_10(call_block, obs_shape, obs_type, true_shape, is_throw):
    batch_size = 8

    x = np.ones((batch_size,) + obs_shape, dtype=np.float32)
    if is_throw:
        with pytest.raises(TFLayerError):
            call_block(obs_shape, obs_type, x)
    else:
        y = call_block(obs_shape, obs_type, x)
        assert y.shape == (batch_size,) + true_shape


pattern_time = [
    # ((2, 4, 8), EnvObservationTypes.UNKNOWN, (2 * 4 * 8,)),
    # ((2, 4, 8), EnvObservationTypes.DISCRETE, (2 * 4 * 8,)),
    # ((2, 4, 8), EnvObservationTypes.CONTINUOUS, (2 * 4 * 8,)),
    ((4, 8), EnvObservationTypes.GRAY_2ch, (4, 8, 1)),
    ((4, 8, 1), EnvObservationTypes.GRAY_3ch, (4, 8, 1)),
    ((4, 8, 3), EnvObservationTypes.COLOR, (4, 8, 3)),
]


@pytest.mark.parametrize("obs_shape, obs_type, true_shape", pattern_time)
def test_time_layer_tf(obs_shape, obs_type, true_shape):
    pytest.importorskip("tensorflow")
    _time_layer(call_block_tf, obs_shape, obs_type, true_shape)


def _time_layer(call_block, obs_shape, obs_type, true_shape):
    batch_size = 7
    seq_len = 3

    x = np.ones((batch_size, seq_len) + obs_shape, dtype=np.float32)
    y = call_block(obs_shape, obs_type, x, enable_time_distributed_layer=True)
    assert y.shape == (batch_size, seq_len) + true_shape
