import numpy as np
import pytest

from srl.rl.models.config.mlp_block import MLPBlockConfig


@pytest.mark.parametrize("enable_noisy_dense", [False, True])
def test_torch_mlp(enable_noisy_dense):
    pytest.importorskip("torch")

    config = MLPBlockConfig()
    config.set((64, 32))

    # ---

    import torch

    batch_size = 16
    x = np.ones((batch_size, 256), dtype=np.float32)

    x = torch.tensor(x)
    block = config.create_block_torch(256, enable_noisy_dense=enable_noisy_dense)
    y = block(x)
    y = y.detach().numpy()

    assert y.shape == (batch_size, 32)
    assert block.out_size == 32


@pytest.mark.parametrize("enable_noisy_dense", [False, True])
def test_torch_dueling_network(enable_noisy_dense):
    pytest.importorskip("torch")
    import torch

    action_num = 5
    dense_units = 32
    batch_size = 16

    from srl.rl.models.config.mlp_block import MLPBlockConfig

    config = MLPBlockConfig()
    config.set_dueling_network((dense_units,))

    block = config.create_block_torch(128, action_num, enable_noisy_dense=enable_noisy_dense)

    x = np.ones((batch_size, 128), dtype=np.float32)
    y = block(torch.tensor(x))
    y = y.detach().numpy()

    assert y.shape == (batch_size, action_num)
