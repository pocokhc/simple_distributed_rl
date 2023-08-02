import numpy as np
import pytest

from srl.rl.models.alphazero_block import AlphaZeroBlockConfig


def test_torch_alphazero_block():
    pytest.importorskip("torch")

    import torch

    config = AlphaZeroBlockConfig()
    config.set_alphazero_block()

    # ---

    batch_size = 16
    x = np.ones((batch_size, 3, 96, 72), dtype=np.float32)

    x = torch.tensor(x)
    block = config.create_block_torch(x.shape[1:])
    y = block(x)
    y = y.detach().numpy()

    assert y.shape == (batch_size, 256, 96, 72)
    assert block.out_shape == (256, 96, 72)


def test_torch_muzero_atari_block():
    pytest.importorskip("torch")

    import torch

    config = AlphaZeroBlockConfig()
    config.set_muzero_atari_block()

    # ---

    batch_size = 16
    x = np.ones((batch_size, 3, 96, 72), dtype=np.float32)

    x = torch.tensor(x)
    block = config.create_block_torch(x.shape[1:])
    y = block(x)
    y = y.detach().numpy()

    assert y.shape == (batch_size, 256, 6, 5)
    assert block.out_shape == (256, 6, 5)
