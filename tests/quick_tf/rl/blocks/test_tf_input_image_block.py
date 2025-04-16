import numpy as np
import pytest

from srl.base.define import SpaceTypes
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.space import SpaceBase
from srl.rl.models.config.input_image_block import InputImageBlockConfig


@pytest.mark.parametrize("rnn", [False, True])
@pytest.mark.parametrize(
    "in_space, out_size",
    [
        [BoxSpace((5, 4)), 5 * 4],  # value block
        [BoxSpace((64, 64), stype=SpaceTypes.GRAY_2ch), 4096],
        [BoxSpace((64, 64, 1), stype=SpaceTypes.GRAY_3ch), 4096],
        [BoxSpace((64, 64, 3), stype=SpaceTypes.COLOR), 4096],
        [BoxSpace((64, 64, 9), stype=SpaceTypes.IMAGE), 4096],
    ],
)
def test_create_block_out_value(in_space: SpaceBase, out_size, rnn):
    pytest.importorskip("tensorflow")
    import tensorflow as tf

    batch_size = 3
    timesteps = 7

    config = InputImageBlockConfig()

    if not in_space.is_image():
        # image以外は例外
        with pytest.raises(AssertionError):
            block = config.create_tf_block(in_space=in_space, rnn=rnn)
        return

    block = config.create_tf_block(in_space=in_space, rnn=rnn)
    print(block)

    # --- shape
    in_data = block.create_dummy_data(np.float32, batch_size=batch_size, timesteps=timesteps)
    in_shape = in_data.shape
    if rnn:
        assert in_shape == (batch_size, timesteps) + in_space.shape
    else:
        assert in_shape == (batch_size,) + in_space.shape

    # --- single data
    x = in_space.sample()
    x = block.to_tf_one_batch(x, tf.float32)
    y = block(x)
    assert y is not None
    print(y.shape)
    if rnn:
        assert y.numpy().shape == (1, 1, out_size)
    else:
        assert y.numpy().shape == (1, out_size)

    # --- batch data
    if rnn:
        x = [[in_space.sample() for _ in range(timesteps)] for _ in range(batch_size)]
    else:
        x = [in_space.sample() for _ in range(batch_size)]
    x = block.to_tf_batches(x, tf.float32)
    y = block(x)
    assert y is not None
    print(y.shape)
    if rnn:
        assert y.numpy().shape == (batch_size, timesteps, out_size)
    else:
        assert y.numpy().shape == (batch_size, out_size)


@pytest.mark.parametrize("rnn", [False, True])
@pytest.mark.parametrize(
    "in_space, out_shape",
    [
        [BoxSpace((64, 64), stype=SpaceTypes.GRAY_2ch), (8, 8, 64)],
        [BoxSpace((64, 64, 1), stype=SpaceTypes.GRAY_3ch), (8, 8, 64)],
        [BoxSpace((64, 64, 3), stype=SpaceTypes.COLOR), (8, 8, 64)],
        [BoxSpace((64, 64, 9), stype=SpaceTypes.IMAGE), (8, 8, 64)],
    ],
)
def test_create_block_out_image(in_space, out_shape, rnn):
    pytest.importorskip("tensorflow")

    batch_size = 3
    seq_size = 5

    config = InputImageBlockConfig()
    block = config.create_tf_block(in_space=in_space, out_flatten=False, rnn=rnn)
    print(block)

    if rnn:
        x = np.ones(
            (seq_size, batch_size) + in_space.shape,
            dtype=np.float32,
        )
    else:
        x = np.ones((batch_size,) + in_space.shape, dtype=np.float32)
    y = block(x)
    assert y is not None
    if rnn:
        assert y.numpy().shape == (seq_size, batch_size) + out_shape
    else:
        assert y.numpy().shape == (batch_size,) + out_shape


@pytest.mark.parametrize("name", ["dqn", "r2d3", "alphazero", "muzero_atari"])
@pytest.mark.parametrize("rnn", [False, True])
def test_tf_image(name, rnn):
    pytest.importorskip("tensorflow")

    cfg = InputImageBlockConfig()
    if name == "dqn":
        cfg.set_dqn_block()
        in_shape = (64, 75, 19)
        out_shape = (8, 10, 64)
    elif name == "r2d3":
        cfg.set_r2d3_block()
        in_shape = (96, 72, 3)
        out_shape = (12, 9, 32)
    elif name == "alphazero":
        cfg.set_alphazero_block()
        in_shape = (96, 72, 3)
        out_shape = (96, 72, 256)
        if rnn:
            pytest.skip("TODO")
    elif name == "muzero_atari":
        in_shape = (96, 72, 3)
        out_shape = (12, 9, 64)
        if rnn:
            pytest.skip("TODO")
    else:
        raise ValueError(name)

    batch_size = 16
    if rnn:
        seq_len = 7
        in_shape2 = (seq_len, batch_size) + in_shape
        out_shape2 = (seq_len, batch_size) + out_shape
    else:
        in_shape2 = (batch_size,) + in_shape
        out_shape2 = (batch_size,) + out_shape

    x = np.ones(in_shape2, dtype=np.float32)
    in_space = BoxSpace(in_shape, stype=SpaceTypes.IMAGE)
    block = cfg.create_tf_block(in_space=in_space, out_flatten=False, rnn=rnn)
    y = block(x)
    assert y is not None
    y = y.numpy()
    assert y.shape == out_shape2

    block.summary()
