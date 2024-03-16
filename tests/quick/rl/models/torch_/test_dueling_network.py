import numpy as np
import pytest


def test_call_torch():
    pytest.importorskip("torch")
    import torch

    action_num = 5
    dense_units = 32
    batch_size = 16

    from srl.rl.models.dueling_network import DuelingNetworkConfig

    config = DuelingNetworkConfig()
    config.set((dense_units,), True)

    block = config.create_block_torch(128, action_num)

    x = np.ones((batch_size, 128), dtype=np.float32)
    y = block(torch.tensor(x))
    y = y.detach().numpy()

    assert y.shape == (batch_size, action_num)


def test_call_torch_lstm():
    pytest.importorskip("torch")
    import torch

    action_num = 5
    dense_units = 32
    batch_size = 16
    seq_len = 7

    from srl.rl.models.dueling_network import DuelingNetworkConfig

    config = DuelingNetworkConfig()
    config.set((dense_units,), True)

    block = config.create_block_torch(128, action_num, enable_time_distributed_layer=True)

    x = np.ones((batch_size, seq_len, 128), dtype=np.float32)
    y = block(torch.tensor(x))
    y = y.detach().numpy()

    assert y.shape == (batch_size, seq_len, action_num)


def test_noisy():
    pytest.importorskip("torch")
    import torch

    action_num = 5
    dense_units = 32
    batch_size = 16

    from srl.rl.models.dueling_network import DuelingNetworkConfig

    config = DuelingNetworkConfig()
    config.set((dense_units,), True)

    block = config.create_block_torch(128, action_num, enable_noisy_dense=True)

    x = np.ones((batch_size, 128), dtype=np.float32)
    y = block(torch.tensor(x))
    y = y.detach().numpy()

    assert y.shape == (batch_size, action_num)


def test_disable():
    pytest.importorskip("torch")
    import torch

    action_num = 5
    dense_units = 32
    batch_size = 16

    from srl.rl.models.dueling_network import DuelingNetworkConfig

    config = DuelingNetworkConfig()
    config.set((dense_units,), False)

    block = config.create_block_torch(128, action_num)

    x = np.ones((batch_size, 128), dtype=np.float32)
    y = block(torch.tensor(x))
    y = y.detach().numpy()

    assert y.shape == (batch_size, action_num)


def test_disable_lstm():
    pytest.importorskip("torch")
    import torch

    action_num = 5
    dense_units = 32
    batch_size = 16
    seq_len = 7

    from srl.rl.models.dueling_network import DuelingNetworkConfig

    config = DuelingNetworkConfig()
    config.set((dense_units,), False)

    block = config.create_block_torch(128, action_num, enable_time_distributed_layer=True)

    x = np.ones((batch_size, seq_len, 128), dtype=np.float32)
    y = block(torch.tensor(x))
    y = y.detach().numpy()

    assert y.shape == (batch_size, seq_len, action_num)
