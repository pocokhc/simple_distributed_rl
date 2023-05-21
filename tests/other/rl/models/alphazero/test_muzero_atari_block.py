import numpy as np
import pytest

from srl.rl.models.alphazero.muzero_atari_block_config import MuzeroAtariBlockConfig


def test_call_tf():
    pytest.importorskip("tensorflow")

    config = MuzeroAtariBlockConfig()
    batch_size = 16
    x = np.ones((batch_size, 96, 72, 3), dtype=np.float32)

    block = config.create_block_tf()
    y = block(x)
    assert y is not None
    y = y.numpy()

    assert y.shape == (batch_size, 6, 5, 256)


def test_call_torch():
    pytest.importorskip("torch")

    import torch

    config = MuzeroAtariBlockConfig()
    batch_size = 16
    x = np.ones((batch_size, 3, 96, 72), dtype=np.float32)

    x = torch.tensor(x)
    block = config.create_block_torch(x.shape[1:])
    y = block(x)
    y = y.detach().numpy()

    assert y.shape == (batch_size, 256, 6, 5)
    assert block.out_shape == (256, 6, 5)
