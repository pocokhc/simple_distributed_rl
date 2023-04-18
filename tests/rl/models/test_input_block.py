import numpy as np
import pytest

from srl.base.define import EnvObservationType
from srl.utils.common import is_package_installed


def call_block_tf(
    obs_shape,
    obs_type,
    x,
    enable_time_distributed_layer=False,
):
    from srl.rl.models.tf.input_block import InputBlock

    block = InputBlock(
        obs_shape,
        obs_type,
        enable_time_distributed_layer=enable_time_distributed_layer,
    )
    y = block(x).numpy()
    return y, block.use_image_layer, None


def call_block_torch(
    obs_shape,
    obs_type,
    x,
    enable_time_distributed_layer=False,
):
    import torch

    from srl.rl.models.torch_.input_block import InputBlock

    block = InputBlock(
        obs_shape,
        obs_type,
        enable_time_distributed_layer=enable_time_distributed_layer,
    )
    y = block(torch.tensor(x))
    out_shape = block.out_shape

    if len(y.shape) == 4:
        # (batch, ch, h, w) -> (batch, h, w, ch)
        y = y.permute((0, 2, 3, 1))
        out_shape = (out_shape[1], out_shape[2], out_shape[0])
    if len(y.shape) == 5:
        # (batch, len, ch, h, w) -> (batch, len, h, w, ch)
        y = y.permute((0, 1, 3, 4, 2))
        out_shape = (out_shape[1], out_shape[2], out_shape[0])

    y = y.detach().numpy()
    return y, block.use_image_layer, out_shape


pattern0 = [
    ((2, 4, 8), EnvObservationType.UNKNOWN, (2 * 4 * 8,), False),
    ((2, 4, 8), EnvObservationType.DISCRETE, (2 * 4 * 8,), False),
    ((2, 4, 8), EnvObservationType.CONTINUOUS, (2 * 4 * 8,), False),
    ((4, 8), EnvObservationType.GRAY_2ch, (4, 8, 1), True),
    ((4, 8, 1), EnvObservationType.GRAY_3ch, (4, 8, 1), True),
    ((4, 8, 3), EnvObservationType.COLOR, (4, 8, 3), True),
    ((4, 8), EnvObservationType.SHAPE2, (4, 8, 1), True),
    ((10, 4, 8), EnvObservationType.SHAPE3, (4, 8, 10), True),
]


@pytest.mark.skipif(not is_package_installed("tensorflow"), reason="no module")
@pytest.mark.parametrize("obs_shape, obs_type, true_shape, true_image", pattern0)
def test_window_0_tf(obs_shape, obs_type, true_shape, true_image):
    _window_0(call_block_tf, obs_shape, obs_type, true_shape, true_image)


@pytest.mark.skipif(not is_package_installed("torch"), reason="no module")
@pytest.mark.parametrize("obs_shape, obs_type, true_shape, true_image", pattern0)
def test_window_0_torch(obs_shape, obs_type, true_shape, true_image):
    _window_0(call_block_torch, obs_shape, obs_type, true_shape, true_image)


def _window_0(call_block, obs_shape, obs_type, true_shape, true_image):
    batch_size = 8

    x = np.ones((batch_size,) + obs_shape, dtype=np.float32)
    y, use_image_layer, out_shape = call_block(obs_shape, obs_type, x)
    assert use_image_layer == true_image
    assert y.shape == (batch_size,) + true_shape
    if out_shape is not None:
        assert out_shape == true_shape


pattern10 = [
    ((10, 2, 4, 8), EnvObservationType.UNKNOWN, (10 * 2 * 4 * 8,), False, False),
    ((10, 2, 4, 8), EnvObservationType.DISCRETE, (10 * 2 * 4 * 8,), False, False),
    ((10, 2, 4, 8), EnvObservationType.CONTINUOUS, (10 * 2 * 4 * 8,), False, False),
    ((10, 4, 8), EnvObservationType.GRAY_2ch, (4, 8, 10), True, False),
    ((10, 4, 8, 1), EnvObservationType.GRAY_3ch, (4, 8, 10), True, False),
    ((10, 4, 8, 3), EnvObservationType.COLOR, None, None, True),
    ((10, 4, 8), EnvObservationType.SHAPE2, (4, 8, 10), True, False),
    ((10, 10, 4, 8), EnvObservationType.SHAPE3, None, None, True),
]


@pytest.mark.skipif(not is_package_installed("tensorflow"), reason="no module")
@pytest.mark.parametrize("obs_shape, obs_type, true_shape, true_image, is_throw", pattern10)
def test_window_10_tf(obs_shape, obs_type, true_shape, true_image, is_throw):
    _window_10(call_block_tf, obs_shape, obs_type, true_shape, true_image, is_throw)


@pytest.mark.skipif(not is_package_installed("torch"), reason="no module")
@pytest.mark.parametrize("obs_shape, obs_type, true_shape, true_image, is_throw", pattern10)
def test_window_10_torch(obs_shape, obs_type, true_shape, true_image, is_throw):
    _window_10(call_block_torch, obs_shape, obs_type, true_shape, true_image, is_throw)


def _window_10(call_block, obs_shape, obs_type, true_shape, true_image, is_throw):
    batch_size = 8

    x = np.ones((batch_size,) + obs_shape, dtype=np.float32)
    if is_throw:
        with pytest.raises(ValueError):
            call_block(obs_shape, obs_type, x)
    else:
        y, use_image_layer, out_shape = call_block(obs_shape, obs_type, x)
        assert use_image_layer == true_image
        assert y.shape == (batch_size,) + true_shape
        if out_shape is not None:
            assert out_shape == true_shape


pattern_time = [
    ((2, 4, 8), EnvObservationType.UNKNOWN, (2 * 4 * 8,), False),
    ((2, 4, 8), EnvObservationType.DISCRETE, (2 * 4 * 8,), False),
    ((2, 4, 8), EnvObservationType.CONTINUOUS, (2 * 4 * 8,), False),
    ((4, 8), EnvObservationType.GRAY_2ch, (4, 8, 1), True),
    ((4, 8, 1), EnvObservationType.GRAY_3ch, (4, 8, 1), True),
    ((4, 8, 3), EnvObservationType.COLOR, (4, 8, 3), True),
    ((4, 8), EnvObservationType.SHAPE2, (4, 8, 1), True),
    ((10, 4, 8), EnvObservationType.SHAPE3, (4, 8, 10), True),
]


@pytest.mark.skipif(not is_package_installed("tensorflow"), reason="no module")
@pytest.mark.parametrize("obs_shape, obs_type, true_shape, true_image", pattern_time)
def test_time_layer_tf(obs_shape, obs_type, true_shape, true_image):
    _time_layer(call_block_tf, obs_shape, obs_type, true_shape, true_image)


@pytest.mark.skipif(not is_package_installed("torch"), reason="no module")
@pytest.mark.parametrize("obs_shape, obs_type, true_shape, true_image", pattern_time)
def test_time_layer_torch(obs_shape, obs_type, true_shape, true_image):
    _time_layer(call_block_torch, obs_shape, obs_type, true_shape, true_image)


def _time_layer(call_block, obs_shape, obs_type, true_shape, true_image):
    batch_size = 7
    seq_len = 3

    x = np.ones((batch_size, seq_len) + obs_shape, dtype=np.float32)
    y, use_image_layer, out_shape = call_block(obs_shape, obs_type, x, enable_time_distributed_layer=True)
    assert use_image_layer == true_image
    assert y.shape == (batch_size, seq_len) + true_shape
    if out_shape is not None:
        assert out_shape == true_shape
