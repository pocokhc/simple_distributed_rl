import numpy as np
import pytest

from srl.rl.models.config.mlp_block import MLPBlockConfig


def test_tf_mlp():
    pytest.importorskip("tensorflow")

    config = MLPBlockConfig()
    config.set_mlp((64, 32))

    # ---

    batch_size = 16
    x = np.ones((batch_size, 256), dtype=np.float32)

    block = config.create_block_tf()
    y = block(x)
    assert y is not None
    y = y.numpy()
    assert y.shape == (batch_size, 32)

    block.summary()


def test_tf_mlp_rnn():
    pytest.importorskip("tensorflow")

    config = MLPBlockConfig()
    config.set_mlp((64, 32))

    # ---

    batch_size = 16
    seq_size = 8
    x = np.ones((batch_size, seq_size, 256), dtype=np.float32)

    block = config.create_block_tf()
    y = block(x)
    assert y is not None
    y = y.numpy()
    assert y.shape == (batch_size, seq_size, 32)

    block.summary()
