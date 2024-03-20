import numpy as np
import pytest

from srl.rl.models.config.image_block import ImageBlockConfig


@pytest.mark.parametrize("name", ["dqn", "r2d3", "alphazero", "muzero_atari"])
def test_torch_image(name):
    pytest.importorskip("torch")

    import torch

    config = ImageBlockConfig()
    if name == "dqn":
        config.set_dqn_block()
        in_shape = (19, 64, 75)
        out_shape = (64, 9, 10)
    elif name == "r2d3":
        config.set_r2d3_block()
        in_shape = (3, 96, 72)
        out_shape = (32, 12, 9)
    elif name == "alphazero":
        config.set_alphazero_block()
        in_shape = (3, 96, 72)
        out_shape = (256, 96, 72)
    elif name == "muzero_atari":
        in_shape = (3, 96, 72)
        out_shape = (64, 13, 10)
    else:
        raise ValueError(name)

    batch_size = 16
    in_shape2 = (batch_size,) + in_shape
    out_shape2 = (batch_size,) + out_shape

    x = np.ones(in_shape2, dtype=np.float32)
    x = torch.tensor(x)
    block = config.create_block_torch(x.shape[1:])
    y = block(x)
    y = y.detach().numpy()

    assert y.shape == out_shape2
    assert block.out_shape == out_shape
    print(block)
