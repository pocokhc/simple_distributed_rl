import numpy as np
import pytest

from srl.rl.models.config.mlp_block import MLPBlockConfig


def test_mlp():
    pytest.importorskip("tensorflow")

    config = MLPBlockConfig()
    config.set((64, 32))

    # ---

    batch_size = 16

    x = np.ones((batch_size, 256), dtype=np.float32)

    block = config.create_tf_block()
    y = block(x)
    assert y is not None
    y = y.numpy()
    assert y.shape == (batch_size, 32)

    block.summary()
