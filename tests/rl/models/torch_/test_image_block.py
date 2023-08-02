import numpy as np
import pytest

from srl.rl.models.image_block import ImageBlockConfig


def test_torch_dqn_image():
    pytest.importorskip("torch")

    import torch

    config = ImageBlockConfig()
    config.set_dqn_image()

    # ---

    batch_size = 16
    x = np.ones((batch_size, 19, 64, 75), dtype=np.float32)

    x = torch.tensor(x)
    block = config.create_block_torch(x.shape[1:], enable_time_distributed_layer=False)
    y = block(x)
    y = y.detach().numpy()

    assert y.shape == (batch_size, 64, 9, 10)
    assert block.out_shape == (64, 9, 10)


def test_torch_dqn_image_lstm():
    pytest.importorskip("torch")

    import torch

    config = ImageBlockConfig()
    config.set_dqn_image()

    # ---

    batch_size = 16
    seq_len = 7
    x = np.ones((batch_size, seq_len, 19, 64, 75), dtype=np.float32)

    x = torch.tensor(x)
    block = config.create_block_torch(x.shape[1:], enable_time_distributed_layer=True)
    y = block(x)
    y = y.detach().numpy()

    assert y.shape == (batch_size, seq_len, 64, 9, 10)
    assert block.out_shape == (64, 9, 10)


def test_torch_r2d3_image():
    pytest.importorskip("torch")

    import torch

    config = ImageBlockConfig()
    config.set_r2d3_image()

    # ---

    batch_size = 16
    x = np.ones((batch_size, 3, 96, 72), dtype=np.float32)

    x = torch.tensor(x)
    block = config.create_block_torch(x.shape[1:], enable_time_distributed_layer=False)
    y = block(x)
    y = y.detach().numpy()

    assert y.shape == (batch_size, 32, 12, 9)
    assert block.out_shape == (32, 12, 9)


def test_torch_r2d3_image_lstm():
    pytest.importorskip("torch")

    import torch

    config = ImageBlockConfig()
    config.set_r2d3_image()

    # ---

    batch_size = 16
    seq_len = 7
    x = np.ones((batch_size, seq_len, 3, 96, 72), dtype=np.float32)

    x = torch.tensor(x)
    block = config.create_block_torch(x.shape[1:], enable_time_distributed_layer=True)
    y = block(x)
    y = y.detach().numpy()

    assert y.shape == (batch_size, seq_len, 32, 12, 9)
    assert block.out_shape == (32, 12, 9)
