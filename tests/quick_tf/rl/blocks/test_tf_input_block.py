import numpy as np
import pytest

from srl.base.define import SpaceTypes
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.multi import MultiSpace
from srl.base.spaces.space import SpaceBase
from srl.rl.models.config.input_config import RLConfigComponentInput


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
    pytest.importorskip("tensorflow")

    batch_size = 3
    seq_size = 7

    config = RLConfigComponentInput()

    block = config.create_input_block_tf(rnn=rnn, in_space=in_space)
    print(block)

    # --- shape
    in_shape = block.create_batch_shape((seq_size, batch_size))
    if isinstance(in_space, MultiSpace):
        for i in range(in_space.space_size):
            assert isinstance(in_space.spaces[i], BoxSpace)
            assert in_shape[i] == (seq_size, batch_size) + in_space.spaces[i].shape
    elif isinstance(in_space, BoxSpace):
        assert in_shape == (seq_size, batch_size) + in_space.shape
    else:
        raise

    # --- single data
    if rnn:
        x = [block.create_batch_single_data(in_space.sample()) for _ in range(seq_size)]
        if in_space.stype == SpaceTypes.MULTI:
            pytest.skip("TODO")
        else:
            x = np.array(x)
    else:
        x = in_space.sample()
        x = block.create_batch_single_data(x)
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
    pytest.importorskip("tensorflow")

    batch_size = 3
    seq_size = 5

    config = RLConfigComponentInput()
    block = config.create_input_block_tf(image_flatten=False, rnn=rnn, in_space=in_space)
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
        assert (
            y.numpy().shape
            == (
                seq_size,
                batch_size,
            )
            + out_shape
        )
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


@pytest.mark.parametrize("name", ["dqn", "r2d3", "alphazero", "muzero_atari"])
@pytest.mark.parametrize("rnn", [False, True])
def test_tf_image(name, rnn):
    pytest.importorskip("tensorflow")

    config = RLConfigComponentInput()
    img_block = config.input_image_block
    if name == "dqn":
        img_block.set_dqn_block()
        in_shape = (64, 75, 19)
        out_shape = (8, 10, 64)
    elif name == "r2d3":
        img_block.set_r2d3_block()
        in_shape = (96, 72, 3)
        out_shape = (12, 9, 32)
    elif name == "alphazero":
        img_block.set_alphazero_block()
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
    block = config.create_input_block_tf(image_flatten=False, rnn=rnn, in_space=in_space)
    y = block(x)
    assert y is not None
    y = y.numpy()
    assert y.shape == out_shape2

    block.summary()
