import numpy as np
import pytest

from srl.utils.common import is_package_installed


@pytest.mark.skipif(not is_package_installed("tensorflow"), reason="no module")
def test_call_tf():
    from srl.rl.models.tf.dueling_network import DuelingNetworkBlock

    action_num = 5
    dense_units = 32
    batch_size = 16

    block = DuelingNetworkBlock(action_num, dense_units)

    x = np.ones((batch_size, 128), dtype=np.float32)
    y = block(x).numpy()

    assert y.shape == (batch_size, action_num)


@pytest.mark.skipif(not is_package_installed("torch"), reason="no module")
def test_call_torch():
    import torch

    from srl.rl.models.torch_.dueling_network import DuelingNetworkBlock

    action_num = 5
    dense_units = 32
    batch_size = 16

    block = DuelingNetworkBlock(128, action_num, dense_units)

    x = np.ones((batch_size, 128), dtype=np.float32)
    y = block(torch.tensor(x))
    y = y.detach().numpy()

    assert y.shape == (batch_size, action_num)
