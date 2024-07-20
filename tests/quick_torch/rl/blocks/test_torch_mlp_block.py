import numpy as np
import pytest

from srl.rl.models.config.mlp_block import MLPBlockConfig


def test_torch_mlp():
    pytest.importorskip("torch")

    config = MLPBlockConfig()
    config.set((64, 32))

    # ---

    import torch

    batch_size = 16
    x = np.ones((batch_size, 256), dtype=np.float32)

    x = torch.tensor(x)
    block = config.create_block_torch(256)
    y = block(x)
    y = y.detach().numpy()

    assert y.shape == (batch_size, 32)
    assert block.out_size == 32
