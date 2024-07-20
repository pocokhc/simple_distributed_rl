import numpy as np
import pytest

from srl.base.define import SpaceTypes
from srl.base.exception import NotSupportedError
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.multi import MultiSpace
from srl.rl.models.config.input_config import RLConfigComponentInput


@pytest.mark.parametrize(
    "in_space, out_size",
    [
        [BoxSpace((5, 4)), 20],  # value block
        [BoxSpace((64, 64), stype=SpaceTypes.GRAY_2ch), 5184],
        [BoxSpace((64, 64, 1), stype=SpaceTypes.GRAY_3ch), 5184],
        [BoxSpace((64, 64, 3), stype=SpaceTypes.COLOR), 5184],
        [BoxSpace((64, 64, 9), stype=SpaceTypes.IMAGE), 5184],
        # MULTI
        [
            MultiSpace(
                [
                    BoxSpace((5, 4)),
                    BoxSpace((64, 64, 3), stype=SpaceTypes.COLOR),
                ]
            ),
            5 * 4 + 5184,
        ],
    ],
)
def test_create_block_out_value(in_space, out_size):
    pytest.importorskip("torch")

    import torch

    batch_size = 3

    config = RLConfigComponentInput()
    block = config.create_input_block_torch(in_space=in_space)
    print(in_space)
    print(block)
    assert block.out_size == out_size

    if in_space.stype == SpaceTypes.MULTI:
        pytest.skip("TODO")
    else:
        x = np.ones((batch_size,) + in_space.shape, dtype=np.float32)
    y = block(torch.tensor(x))
    y = y.detach().numpy()

    assert y.shape == (3, out_size)


@pytest.mark.parametrize(
    "in_space, out_shape",
    [
        [BoxSpace((64, 64), stype=SpaceTypes.GRAY_2ch), (64, 9, 9)],
        [BoxSpace((64, 64, 1), stype=SpaceTypes.GRAY_3ch), (64, 9, 9)],
        [BoxSpace((64, 64, 3), stype=SpaceTypes.COLOR), (64, 9, 9)],
        [BoxSpace((64, 64, 9), stype=SpaceTypes.IMAGE), (64, 9, 9)],
    ],
)
def test_create_block_out_image(in_space, out_shape):
    pytest.importorskip("torch")

    import torch

    batch_size = 3
    config = RLConfigComponentInput()
    block = config.create_input_block_torch(image_flatten=False, in_space=in_space)
    print(block)
    assert block.out_shape == out_shape

    x = np.ones((batch_size,) + in_space.shape, dtype=np.float32)
    y = block(torch.tensor(x))
    y = y.detach().numpy()

    assert y.shape == (3,) + out_shape


@pytest.mark.parametrize(
    "in_space, out_shape",
    [
        [BoxSpace((64, 64, 3), stype=SpaceTypes.COLOR), (64, 9, 9)],  # image block
    ],
)
def test_create_block_out_multi(in_space, out_shape):
    pytest.skip("TODO")
    pytest.importorskip("torch")

    import torch

    from srl.rl.torch_.blocks.input_block import create_in_block_out_image

    batch_size = 3
    img_conf = ImageBlockConfig()

    with pytest.raises(NotSupportedError):
        create_in_block_out_image(img_conf, BoxSpace((5, 4)))
    with pytest.raises(NotSupportedError):
        create_in_block_out_image(img_conf, MultiSpace([BoxSpace((5, 4))]))

    block = create_in_block_out_image(img_conf, in_space)
    print(block)
    assert block.out_shape == out_shape

    x = np.ones((batch_size,) + in_space.shape, dtype=np.float32)
    y = block(torch.tensor(x))
    y = y.detach().numpy()

    assert y.shape == (3,) + out_shape


@pytest.mark.parametrize("name", ["dqn", "r2d3", "alphazero", "muzero_atari"])
def test_torch_image(name):
    pytest.importorskip("torch")

    import torch

    config = RLConfigComponentInput()
    img_block = config.input_image_block
    if name == "dqn":
        img_block.set_dqn_block()
        in_shape = (64, 75, 19)  # h, w, ch
        out_shape = (64, 9, 10)  # ch, h, w
    elif name == "r2d3":
        img_block.set_r2d3_block()
        in_shape = (96, 72, 3)  # h, w, ch
        out_shape = (32, 12, 9)  # ch, h, w
    elif name == "alphazero":
        img_block.set_alphazero_block()
        in_shape = (96, 72, 3)  # h, w, ch
        out_shape = (256, 96, 72)  # ch, h, w
    elif name == "muzero_atari":
        img_block.set_muzero_atari_block()
        in_shape = (96, 72, 3)  # h, w, ch
        out_shape = (256, 6, 5)  # ch, h, w
    else:
        raise ValueError(name)

    batch_size = 16
    in_shape2 = (batch_size,) + in_shape
    out_shape2 = (batch_size,) + out_shape

    in_space = BoxSpace(in_shape, stype=SpaceTypes.IMAGE)
    block = config.create_input_block_torch(image_flatten=False, in_space=in_space)
    x = np.ones(in_shape2, dtype=np.float32)
    x = torch.tensor(x)
    y = block(x)
    y = y.detach().numpy()
    print(block)

    assert y.shape == out_shape2
    assert block.out_shape == out_shape
