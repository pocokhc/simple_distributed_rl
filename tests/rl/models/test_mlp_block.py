import numpy as np
import pytest

from srl.rl.models.mlp_block_config import MLPBlockConfig
from srl.utils.common import is_package_installed


@pytest.mark.skipif(not is_package_installed("tensorflow"), reason="no module")
def test_call_tf():
    config = MLPBlockConfig((64, 32))
    batch_size = 16
    x = np.ones((batch_size, 256), dtype=np.float32)

    block = config.create_block_tf()
    y = block(x).numpy()

    assert y.shape == (batch_size, 32)


@pytest.mark.skipif(not is_package_installed("torch"), reason="no module")
def test_call_torch():
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
