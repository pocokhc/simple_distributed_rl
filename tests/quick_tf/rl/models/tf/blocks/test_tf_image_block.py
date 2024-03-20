import numpy as np
import pytest

from srl.rl.models.config.image_block import ImageBlockConfig


@pytest.mark.parametrize("name", ["dqn", "r2d3", "alphazero", "muzero_atari"])
@pytest.mark.parametrize("rnn", [False, True])
def test_tf_image(name, rnn):
    pytest.importorskip("tensorflow")

    config = ImageBlockConfig()
    if name == "dqn":
        config.set_dqn_block()
        in_shape = (64, 75, 19)
        out_shape = (8, 10, 64)
    elif name == "r2d3":
        config.set_r2d3_block()
        in_shape = (96, 72, 3)
        out_shape = (12, 9, 32)
    elif name == "alphazero":
        config.set_alphazero_block()
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
    block = config.create_block_tf(enable_rnn=rnn)
    y = block(x)
    assert y is not None
    y = y.numpy()
    assert y.shape == out_shape2

    block.summary()
