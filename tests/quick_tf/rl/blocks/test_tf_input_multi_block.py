import numpy as np
import pytest

from srl.base.define import SpaceTypes
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.multi import MultiSpace
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
        # MULTI
        [
            MultiSpace(
                [
                    BoxSpace((5, 4)),
                    BoxSpace((64, 64, 3), stype=SpaceTypes.COLOR),
                ]
            ),
            5 * 4 + 4096,
        ],
    ],
)
def test_create_block_out_value(in_space: SpaceBase, out_size, rnn):
    pytest.skip("TODO")
    pytest.importorskip("tensorflow")
    import tensorflow as tf

    batch_size = 3
    seq_size = 7

    config = InputImageBlockConfig()

    block = config.create_tf_block(in_space=in_space, rnn=rnn)
    print(block)

    # --- shape
    in_data = block.create_dummy_data(np.float32, batch_size=batch_size, timesteps=seq_size)
    in_shape = in_data.shape
    if isinstance(in_space, MultiSpace):
        for i in range(in_space.space_size):
            assert isinstance(in_space.spaces[i], BoxSpace)
            assert in_shape[i] == (seq_size, batch_size) + in_space.spaces[i].shape
    elif isinstance(in_space, BoxSpace):
        assert in_shape == (seq_size, batch_size) + in_space.shape
    else:
        raise

    # --- single data
    x = block.to_tf_one_batch(in_space.sample(), tf.float32)
    y = block(x)
    assert y is not None
    print(y.shape)
    if rnn:
        assert y.numpy().shape == (seq_size, 1, out_size)
    else:
        assert y.numpy().shape == (1, out_size)

    # --- batch data
    if rnn:
        x = [[in_space.sample() for _ in range(batch_size)] for _ in range(seq_size)]
        x = block.create_batch_stack_data(x)
    else:
        x = [in_space.sample() for _ in range(batch_size)]
        x = block.create_batch_stack_data(x)
    y = block(x)
    assert y is not None
    print(y.shape)
    if rnn:
        assert y.numpy().shape == (seq_size, batch_size, out_size)
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
    pytest.skip("TODO")
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


@pytest.mark.parametrize(
    "in_space, out_shape",
    [
        [BoxSpace((64, 64, 3), stype=SpaceTypes.COLOR), (64, 9, 9)],  # image block
    ],
)
def test_create_block_out_multi(in_space, out_shape):
    pytest.skip("TODO")
