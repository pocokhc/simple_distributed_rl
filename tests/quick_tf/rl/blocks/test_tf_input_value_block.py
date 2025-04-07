import numpy as np
import pytest

from srl.base.define import SpaceTypes
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.space import SpaceBase
from srl.rl.models.config.input_value_block import InputValueBlockConfig


@pytest.mark.parametrize("rnn", [False, True])
@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize(
    "in_space, out_size",
    [
        [BoxSpace((5, 4)), 5 * 4],  # value block
        [BoxSpace((64, 64), stype=SpaceTypes.GRAY_2ch), 64 * 64],
        [BoxSpace((64, 64, 1), stype=SpaceTypes.GRAY_3ch), 64 * 64],
        [BoxSpace((64, 64, 3), stype=SpaceTypes.COLOR), 64 * 64 * 3],
        [BoxSpace((64, 64, 9), stype=SpaceTypes.IMAGE), 64 * 64 * 9],
    ],
)
def test_create_block_out_value(in_space: SpaceBase, out_size, flatten, rnn):
    pytest.importorskip("tensorflow")
    import tensorflow as tf

    batch_size = 3
    timesteps = 7

    cfg = InputValueBlockConfig()
    block = cfg.create_tf_block(in_space=in_space, input_flatten=flatten, rnn=rnn)
    print(block)

    # --- shape
    in_data = block.create_dummy_data(np.float32, batch_size=batch_size, timesteps=timesteps)
    in_shape = in_data.shape

    if rnn:
        assert in_shape == (batch_size, timesteps) + in_space.np_shape
    else:
        assert in_shape == (batch_size,) + in_space.np_shape

    # --- single data
    if rnn:
        x = [in_space.sample() for _ in range(timesteps)]
    else:
        x = in_space.sample()
    x = block.to_tf_one_batch(x, tf.float32)
    y = block(x)
    assert y is not None
    print(y.shape)
    if flatten:
        if rnn:
            assert y.numpy().shape == (1, timesteps, out_size)
        else:
            assert y.numpy().shape == (1, out_size)
    else:
        if rnn:
            assert y.numpy().shape == (1, timesteps) + in_space.np_shape
        else:
            assert y.numpy().shape == (1,) + in_space.np_shape

    # --- batch data
    if rnn:
        x = [[in_space.sample() for _ in range(timesteps)] for _ in range(batch_size)]
    else:
        x = [in_space.sample() for _ in range(batch_size)]
    x = block.to_tf_batches(x, tf.float32)
    y = block(x)
    assert y is not None
    print(y.shape)
    if flatten:
        if rnn:
            assert y.numpy().shape == (batch_size, timesteps, out_size)
        else:
            assert y.numpy().shape == (batch_size, out_size)
    else:
        if rnn:
            assert y.numpy().shape == (batch_size, timesteps) + in_space.np_shape
        else:
            assert y.numpy().shape == (batch_size,) + in_space.np_shape
