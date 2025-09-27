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
    pytest.importorskip("tensorflow")

    batch_size = 3
    timesteps = 7

    cfg = InputValueBlockConfig()
    block = cfg.create_tf_block(rnn=rnn)
    print(block)

    # --- shape
    if rnn:
        in_shape = (batch_size, timesteps) + in_space.shape
    else:
        in_shape = (batch_size,) + in_space.shape

    # --- batch data
    if rnn:
        x = [[in_space.sample() for _ in range(timesteps)] for _ in range(batch_size)]
    else:
        x = [in_space.sample() for _ in range(batch_size)]
    x = np.asarray(x, dtype=np.float32)
    assert x.shape == in_shape
    y = block(x)
    assert y is not None
    print(y.shape)
    if rnn:
        assert y.numpy().shape == (batch_size, timesteps, out_size)
    else:
        assert y.numpy().shape == (batch_size, out_size)
