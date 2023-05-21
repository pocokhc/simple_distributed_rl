import numpy as np
import pytest

from srl.rl.models.mlp import MLPBlockConfig


def test_call_tf():
    pytest.importorskip("tensorflow")

    config = MLPBlockConfig((64, 32))
    batch_size = 16
    x = np.ones((batch_size, 256), dtype=np.float32)

    block = config.create_block_tf()
    y = block(x)
    assert y is not None
    y = y.numpy()
    assert y.shape == (batch_size, 32)


def test_call_torch():
    pytest.importorskip("torch")

    import torch

    config = MLPBlockConfig((64, 32))
    batch_size = 16
    x = np.ones((batch_size, 256), dtype=np.float32)

    x = torch.tensor(x)
    block = config.create_block_torch(256)
    y = block(x)
    y = y.detach().numpy()

    assert y.shape == (batch_size, 32)
    assert block.out_shape == (32,)
