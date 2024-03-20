import numpy as np
import pytest

from srl.rl.models.config.mlp_block import MLPBlockConfig


@pytest.mark.parametrize("rnn", [False, True])
@pytest.mark.parametrize("enable_noisy_dense", [False, True])
def test_mlp(rnn, enable_noisy_dense):
    pytest.importorskip("tensorflow")

    config = MLPBlockConfig()
    config.set((64, 32))

    # ---

    batch_size = 16
    seq_size = 4

    if rnn:
        x = np.ones((seq_size, batch_size, 256), dtype=np.float32)
    else:
        x = np.ones((batch_size, 256), dtype=np.float32)

    block = config.create_block_tf(enable_rnn=rnn, enable_noisy_dense=enable_noisy_dense)
    y = block(x)
    assert y is not None
    y = y.numpy()
    if rnn:
        assert y.shape == (seq_size, batch_size, 32)
    else:
        assert y.shape == (batch_size, 32)

    block.summary()


@pytest.mark.parametrize("rnn", [False, True])
@pytest.mark.parametrize("enable_noisy_dense", [False, True])
def test_dueling_network(rnn, enable_noisy_dense):
    pytest.importorskip("tensorflow")

    action_num = 5
    dense_units = 32
    batch_size = 16
    seq_size = 4

    from srl.rl.models.config.mlp_block import MLPBlockConfig

    config = MLPBlockConfig()
    config.set_dueling_network((dense_units,))

    block = config.create_block_tf(action_num, enable_rnn=rnn, enable_noisy_dense=enable_noisy_dense)

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
