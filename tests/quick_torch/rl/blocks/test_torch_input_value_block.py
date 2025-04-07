import pytest

from srl.base.define import SpaceTypes
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.space import SpaceBase
from srl.rl.models.config.input_value_block import InputValueBlockConfig


@pytest.mark.parametrize("rnn", [False, True])
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
def test_create_block_out_value(in_space: SpaceBase, out_size, rnn):
    pytest.importorskip("torch")
    import torch

    batch_size = 3
    timesteps = 7

    cfg = InputValueBlockConfig()
    block = cfg.create_torch_block(in_space.np_shape, input_flatten=True, reshape_for_rnn=rnn)
    print(in_space)
    print(block)
    assert block.out_size == out_size

    # --- single data
    if rnn:
        x = [in_space.sample() for _ in range(timesteps)]
    else:
        x = in_space.sample()
    x = block.to_torch_one_batch(x, "cpu", torch.float32)
    y = block(x)
    if rnn:
        assert y.detach().numpy().shape == (timesteps, out_size)

        y = block.unreshape_for_rnn(y)

        y = y.detach().numpy()
        assert y.shape == (1, timesteps, out_size)
    else:
        y = y.detach().numpy()
        assert y.shape == (1, out_size)

    # --- batch
    if rnn:
        x = [[in_space.sample() for _ in range(timesteps)] for _ in range(batch_size)]
    else:
        x = [in_space.sample() for _ in range(batch_size)]
    x = block.to_torch_batches(x, "cpu", torch.float32)
    y = block(x)
    if rnn:
        assert y.detach().numpy().shape == (batch_size * timesteps, out_size)

        y = block.unreshape_for_rnn(y)

        y = y.detach().numpy()
        assert y.shape == (batch_size, timesteps, out_size)
    else:
        y = y.detach().numpy()
        assert y.shape == (batch_size, out_size)
