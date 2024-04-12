import numpy as np
import pytest

from srl.base.define import SpaceTypes
from srl.base.exception import NotSupportedError
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.multi import MultiSpace
from srl.rl.models.config.image_block import ImageBlockConfig
from srl.rl.models.config.mlp_block import MLPBlockConfig


@pytest.mark.parametrize(
    "in_space, out_size",
    [
        [BoxSpace((5, 4)), 512],  # value block
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
            512 + 5184,
        ],
    ],
)
def test_create_block_out_value(in_space, out_size):
    pytest.importorskip("torch")

    import torch

    from srl.rl.torch_.blocks.input_block import create_in_block_out_value

    batch_size = 3

    mlp_conf = MLPBlockConfig()
    mlp_conf.set()
    img_conf = ImageBlockConfig()
    block = create_in_block_out_value(mlp_conf, img_conf, in_space)
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
