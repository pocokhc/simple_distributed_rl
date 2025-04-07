import numpy as np
import pytest

from srl.rl.models.config.dueling_network import DuelingNetworkConfig


@pytest.mark.parametrize("rnn", [False, True])
@pytest.mark.parametrize("enable_noisy_dense", [False, True])
def test_mlp(rnn, enable_noisy_dense):
    pytest.importorskip("tensorflow")

    config = DuelingNetworkConfig()
    config.set((64, 32))

    # ---

    action_num = 5
    batch_size = 16
    seq_size = 4

    if rnn:
        x = np.ones((seq_size, batch_size, 256), dtype=np.float32)
    else:
        x = np.ones((batch_size, 256), dtype=np.float32)

    block = config.create_tf_block(action_num, rnn=rnn, enable_noisy_dense=enable_noisy_dense)
    y = block(x)
    assert y is not None
    y = y.numpy()
    if rnn:
        assert y.shape == (seq_size, batch_size, action_num)
    else:
        assert y.shape == (batch_size, action_num)

    block.summary()


@pytest.mark.parametrize("rnn", [False, True])
@pytest.mark.parametrize("enable_noisy_dense", [False, True])
def test_dueling_network(rnn, enable_noisy_dense):
    pytest.importorskip("tensorflow")

    action_num = 5
    dense_units = 32
    batch_size = 16
    seq_size = 4

    config = DuelingNetworkConfig()
    config.set_dueling_network((dense_units,))

    block = config.create_tf_block(action_num, rnn=rnn, enable_noisy_dense=enable_noisy_dense)

    if rnn:
        x = np.ones((seq_size, batch_size, 128), dtype=np.float32)
    else:
        x = np.ones((batch_size, 128), dtype=np.float32)
    y = block(x)
    assert y is not None
    y = y.numpy()
    if rnn:
        assert y.shape == (seq_size, batch_size, action_num)
    else:
        assert y.shape == (batch_size, action_num)

    block.summary()
