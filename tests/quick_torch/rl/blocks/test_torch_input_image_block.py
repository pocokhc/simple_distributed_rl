import numpy as np
import pytest

from srl.base.define import SpaceTypes
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.space import SpaceBase
from srl.rl.models.config.input_image_block import InputImageBlockConfig


@pytest.mark.parametrize("rnn", [False, True])
@pytest.mark.parametrize(
    "in_space, out_flatten, out_shape",
    [
        [BoxSpace((5, 4)), True, 20],  # value block
        [BoxSpace((64, 64), stype=SpaceTypes.GRAY_2ch), True, 5184],
        [BoxSpace((64, 64, 1), stype=SpaceTypes.GRAY_3ch), True, 5184],
        [BoxSpace((64, 64, 3), stype=SpaceTypes.COLOR), True, 5184],
        [BoxSpace((64, 64, 9), stype=SpaceTypes.IMAGE), True, 5184],
        #
        [BoxSpace((64, 64), stype=SpaceTypes.GRAY_2ch), False, (64, 9, 9)],
        [BoxSpace((64, 64, 1), stype=SpaceTypes.GRAY_3ch), False, (64, 9, 9)],
        [BoxSpace((64, 64, 3), stype=SpaceTypes.COLOR), False, (64, 9, 9)],
        [BoxSpace((64, 64, 9), stype=SpaceTypes.IMAGE), False, (64, 9, 9)],
    ],
)
def test_create_block_out_value(in_space: SpaceBase, out_flatten, out_shape, rnn):
    pytest.importorskip("torch")
    import torch

    if out_flatten:
        out_size = out_shape

    batch_size = 3
    timesteps = 7

    config = InputImageBlockConfig()

    if not in_space.is_image():
        # image以外は例外
        with pytest.raises(AssertionError):
            block = config.create_torch_block(in_space=in_space, out_flatten=out_flatten, reshape_for_rnn=rnn)
        return

    block = config.create_torch_block(in_space=in_space, out_flatten=out_flatten, reshape_for_rnn=rnn)
    print(in_space)
    print(block)
    if out_flatten:
        assert block.out_size == out_size
    else:
        assert block.out_shape == out_shape

    # --- single data
    if rnn:
        x = [in_space.sample() for _ in range(timesteps)]
    else:
        x = in_space.sample()
    x = block.to_torch_one_batch(x, "cpu", torch.float32)
    y = block(x)
    if rnn:
        if out_flatten:
            assert y.detach().numpy().shape == (timesteps, out_size)
        else:
            assert y.detach().numpy().shape == (timesteps,) + out_shape

        y = block.unreshape_for_rnn(y)

        y = y.detach().numpy()
        if out_flatten:
            assert y.shape == (1, timesteps, out_size)
        else:
            assert y.shape == (1, timesteps) + out_shape
    else:
        y = y.detach().numpy()
        if out_flatten:
            assert y.shape == (1, out_size)
        else:
            assert y.shape == (1,) + out_shape

    # --- batch
    if rnn:
        x = [[in_space.sample() for _ in range(timesteps)] for _ in range(batch_size)]
    else:
        x = [in_space.sample() for _ in range(batch_size)]
    x = block.to_torch_batches(x, "cpu", torch.float32)
    y = block(x)
    if rnn:
        if out_flatten:
            assert y.detach().numpy().shape == (batch_size * timesteps, out_size)
        else:
            assert y.detach().numpy().shape == (batch_size * timesteps,) + out_shape

        y = block.unreshape_for_rnn(y)

        y = y.detach().numpy()
        if out_flatten:
            assert y.shape == (batch_size, timesteps, out_size)
        else:
            assert y.shape == (batch_size, timesteps) + out_shape
    else:
        y = y.detach().numpy()
        if out_flatten:
            assert y.shape == (batch_size, out_size)
        else:
            assert y.shape == (batch_size,) + out_shape


@pytest.mark.parametrize("name", ["dqn", "r2d3", "alphazero", "muzero_atari"])
def test_torch_image(name):
    pytest.importorskip("torch")

    import torch

    cfg = InputImageBlockConfig()
    if name == "dqn":
        cfg.set_dqn_block()
        in_shape = (64, 75, 19)  # h, w, ch
        out_shape = (64, 9, 10)  # ch, h, w
    elif name == "r2d3":
        cfg.set_r2d3_block()
        in_shape = (96, 72, 3)  # h, w, ch
        out_shape = (32, 12, 9)  # ch, h, w
    elif name == "alphazero":
        cfg.set_alphazero_block()
        in_shape = (96, 72, 3)  # h, w, ch
        out_shape = (256, 96, 72)  # ch, h, w
    elif name == "muzero_atari":
        cfg.set_muzero_atari_block()
        in_shape = (96, 72, 3)  # h, w, ch
        out_shape = (256, 6, 5)  # ch, h, w
    else:
        raise ValueError(name)

    batch_size = 16
    in_shape2 = (batch_size,) + in_shape
    out_shape2 = (batch_size,) + out_shape

    in_space = BoxSpace(in_shape, stype=SpaceTypes.IMAGE)
    block = cfg.create_torch_block(out_flatten=False, in_space=in_space)
    x = np.ones(in_shape2, dtype=np.float32)
    x = torch.tensor(x)
    y = block(x)
    y = y.detach().numpy()
    print(block)

    assert y.shape == out_shape2
    assert block.out_shape == out_shape
