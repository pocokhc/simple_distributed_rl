import numpy as np
import pytest

from srl.rl.models.image_block import ImageBlockConfig


def test_tf_dqn_image():
    pytest.importorskip("tensorflow")

    config = ImageBlockConfig()
    config.set_dqn_image()

    # ---

    batch_size = 16
    x = np.ones((batch_size, 64, 75, 19), dtype=np.float32)

    block = config.create_block_tf(enable_time_distributed_layer=False)
    y = block(x)
    assert y is not None
    y = y.numpy()
    assert y.shape == (batch_size, 8, 10, 64)


def test_tf_dqn_image_lstm():
    pytest.importorskip("tensorflow")

    config = ImageBlockConfig()
    config.set_dqn_image()

    # ---

    batch_size = 16
    seq_len = 7
    x = np.ones((batch_size, seq_len, 64, 75, 19), dtype=np.float32)

    block = config.create_block_tf(enable_time_distributed_layer=True)
    y = block(x)
    assert y is not None
    y = y.numpy()
    assert y.shape == (batch_size, seq_len, 8, 10, 64)


def test_tf_r2d3_image():
    pytest.importorskip("tensorflow")

    config = ImageBlockConfig()
    config.set_r2d3_image()

    # ---

    batch_size = 16
    x = np.ones((batch_size, 96, 72, 3), dtype=np.float32)

    block = config.create_block_tf(enable_time_distributed_layer=False)
    y = block(x)
    assert y is not None
    y = y.numpy()
    assert y.shape == (batch_size, 12, 9, 32)


def test_tf_r2d3_image_lstm():
    pytest.importorskip("tensorflow")

    config = ImageBlockConfig()
    config.set_r2d3_image()

    # ---

    batch_size = 16
    seq_len = 7
    x = np.ones((batch_size, seq_len, 96, 72, 3), dtype=np.float32)

    block = config.create_block_tf(enable_time_distributed_layer=True)
    y = block(x)
    assert y is not None
    y = y.numpy()
    assert y.shape == (batch_size, seq_len, 12, 9, 32)
