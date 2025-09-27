import numpy as np
import pytest

from srl.base.define import SpaceTypes
from srl.base.spaces.box import BoxSpace
from srl.rl.models.config.input_block import InputValueBlockConfig


@pytest.mark.parametrize("rnn", [False, True])
@pytest.mark.parametrize(
    "in_space, out_size",
    [
        [BoxSpace((5, 4)), 5 * 4],  # value block
        [BoxSpace((64, 64), stype=SpaceTypes.GRAY_HW), 64 * 64],
        [BoxSpace((64, 64, 1), stype=SpaceTypes.GRAY_HW1), 64 * 64],
        [BoxSpace((64, 64, 3), stype=SpaceTypes.RGB), 64 * 64 * 3],
        [BoxSpace((64, 64, 9), stype=SpaceTypes.FEATURE_MAP), 64 * 64 * 9],
    ],
)
def test_create_block_out_value(in_space: BoxSpace, out_size, rnn):
    pytest.importorskip("torch")
    import torch

    batch_size = 3
    timesteps = 7

    cfg = InputValueBlockConfig()
    block = cfg.create_torch_block(in_space, reshape_for_rnn=rnn)
    print(in_space)
    print(block)
    assert block.out_size == out_size

    # --- single data
    if rnn:
        x = [in_space.sample() for _ in range(timesteps)]
    else:
        x = in_space.sample()
    x = torch.tensor(np.asarray(x)[np.newaxis, ...], dtype=torch.float32)
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
    x = torch.tensor(np.asarray(x), dtype=torch.float32)
    y = block(x)
    if rnn:
        assert y.detach().numpy().shape == (batch_size * timesteps, out_size)

        y = block.unreshape_for_rnn(y)

        y = y.detach().numpy()
        assert y.shape == (batch_size, timesteps, out_size)
    else:
        y = y.detach().numpy()
        assert y.shape == (batch_size, out_size)
