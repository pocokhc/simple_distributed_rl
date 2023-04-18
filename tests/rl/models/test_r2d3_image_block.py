import numpy as np
import pytest

from srl.rl.models.r2d3_image_block_config import R2D3ImageBlockConfig
from srl.utils.common import is_package_installed


@pytest.mark.skipif(not is_package_installed("tensorflow"), reason="no module")
def test_call_tf():
    config = R2D3ImageBlockConfig()
    batch_size = 16
    x = np.ones((batch_size, 96, 72, 3), dtype=np.float32)

    block = config.create_block_tf()
    y = block(x).numpy()

    assert y.shape == (batch_size, 12, 9, 32)


@pytest.mark.skipif(not is_package_installed("torch"), reason="no module")
def test_call_torch():
    import torch

    config = R2D3ImageBlockConfig()
    batch_size = 16
    x = np.ones((batch_size, 3, 96, 72), dtype=np.float32)

    x = torch.tensor(x)
    block = config.create_block_torch(x.shape[1:])
    y = block(x)
    y = y.detach().numpy()

    assert y.shape == (batch_size, 32, 12, 9)
    assert block.out_shape == (32, 12, 9)


@pytest.mark.skipif(not is_package_installed("tensorflow"), reason="no module")
def test_call_lstm_tf():
    config = R2D3ImageBlockConfig()
    batch_size = 16
    seq_len = 7
    x = np.ones((batch_size, seq_len, 96, 72, 3), dtype=np.float32)

    block = config.create_block_tf(enable_time_distributed_layer=True)
    y = block(x).numpy()

    assert y.shape == (batch_size, seq_len, 12, 9, 32)


@pytest.mark.skipif(not is_package_installed("torch"), reason="no module")
def test_call_lstm_torch():
    import torch

    config = R2D3ImageBlockConfig()
    batch_size = 16
    seq_len = 7
    x = np.ones((batch_size, seq_len, 3, 96, 72), dtype=np.float32)

    x = torch.tensor(x)
    block = config.create_block_torch(x.shape[1:], enable_time_distributed_layer=True)
    y = block(x)
    y = y.detach().numpy()

    assert y.shape == (batch_size, seq_len, 32, 12, 9)
    assert block.out_shape == (32, 12, 9)
