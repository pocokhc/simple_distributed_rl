import numpy as np
import pytest

from srl.rl.models.config.hidden_block import HiddenBlockConfig


def test_torch_mlp():
    pytest.importorskip("torch")

    config = HiddenBlockConfig()
    config.set((64, 32))

    # ---

    import torch

    batch_size = 16
    x = np.ones((batch_size, 256), dtype=np.float32)

    x = torch.tensor(x)
    block = config.create_torch_block(256)
    y = block(x)
    y = y.detach().numpy()

    assert y.shape == (batch_size, 32)
    assert block.out_size == 32
