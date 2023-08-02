import numpy as np
import pytest


def test_call_tf():
    pytest.importorskip("tensorflow")

    from srl.rl.models.tf.dueling_network import DuelingNetworkBlock

    action_num = 5
    dense_units = 32
    batch_size = 16

    block = DuelingNetworkBlock(action_num, dense_units)

    x = np.ones((batch_size, 128), dtype=np.float32)
    y = block(x)
    assert y is not None
    y = y.numpy()
    assert y.shape == (batch_size, action_num)
