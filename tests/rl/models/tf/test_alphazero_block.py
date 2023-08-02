import numpy as np
import pytest

from srl.rl.models.alphazero_block import AlphaZeroBlockConfig


def test_tf_alphazero_block():
    pytest.importorskip("tensorflow")

    config = AlphaZeroBlockConfig()
    config.set_alphazero_block()

    # ---

    batch_size = 16
    x = np.ones((batch_size, 96, 72, 3), dtype=np.float32)

    block = config.create_block_tf()
    y = block(x)
    assert y is not None
    y = y.numpy()

    assert y.shape == (batch_size, 96, 72, 256)


def test_tf_muzero_atari_block():
    pytest.importorskip("tensorflow")

    config = AlphaZeroBlockConfig()
    config.set_muzero_atari_block()

    # ---

    batch_size = 16
    x = np.ones((batch_size, 96, 72, 3), dtype=np.float32)

    block = config.create_block_tf()
    y = block(x)
    assert y is not None
    y = y.numpy()

    assert y.shape == (batch_size, 6, 5, 256)
