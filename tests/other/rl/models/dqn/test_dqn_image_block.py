import numpy as np
import pytest

from srl.rl.models.dqn.dqn_image_block_config import DQNImageBlockConfig
from srl.utils.common import is_package_installed


def test_call_tf():
    pytest.importorskip("tensorflow")

    config = DQNImageBlockConfig()
    batch_size = 16
    x = np.ones((batch_size, 64, 75, 19), dtype=np.float32)

    block = config.create_block_tf()
    y = block(x)
    assert y is not None
    y = y.numpy()
    assert y.shape == (batch_size, 8, 10, 64)


def test_call_torch():
    pytest.importorskip("torch")

    import torch

    config = DQNImageBlockConfig()
    batch_size = 16
    x = np.ones((batch_size, 19, 64, 75), dtype=np.float32)

    x = torch.tensor(x)
    block = config.create_block_torch(x.shape[1:])
    y = block(x)
    y = y.detach().numpy()

    assert y.shape == (batch_size, 64, 9, 10)
    assert block.out_shape == (64, 9, 10)


@pytest.mark.skipif(not is_package_installed("tensorflow"), reason="no module")
def test_call_lstm_tf():
    config = DQNImageBlockConfig()
    batch_size = 16
    seq_len = 7
    x = np.ones((batch_size, seq_len, 64, 75, 19), dtype=np.float32)

    block = config.create_block_tf(enable_time_distributed_layer=True)
    y = block(x)
    assert y is not None
    y = y.numpy()
    assert y.shape == (batch_size, seq_len, 8, 10, 64)


@pytest.mark.skipif(not is_package_installed("torch"), reason="no module")
def test_call_lstm_torch():
    import torch

    config = DQNImageBlockConfig()
    batch_size = 16
    seq_len = 7
    x = np.ones((batch_size, seq_len, 19, 64, 75), dtype=np.float32)

    x = torch.tensor(x)
    block = config.create_block_torch(x.shape[1:], enable_time_distributed_layer=True)
    y = block(x)
    y = y.detach().numpy()

    assert y.shape == (batch_size, seq_len, 64, 9, 10)
    assert block.out_shape == (64, 9, 10)
