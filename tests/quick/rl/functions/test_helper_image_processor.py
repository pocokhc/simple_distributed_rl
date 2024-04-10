import numpy as np
import pytest

from srl.base.define import SpaceTypes
from srl.rl.functions import helper


@pytest.mark.parametrize(
    "img_shape, from_space_type, true_base_shape",
    [
        [(64, 32), SpaceTypes.GRAY_2ch, (64, 32)],
        [(64, 32, 1), SpaceTypes.GRAY_3ch, (64, 32)],
        [(64, 32, 3), SpaceTypes.COLOR, (64, 32)],
    ],
)
@pytest.mark.parametrize("to_space_type", [SpaceTypes.GRAY_2ch, SpaceTypes.GRAY_3ch, SpaceTypes.COLOR])
@pytest.mark.parametrize("resize", [None, (18, 4)])
@pytest.mark.parametrize("trimming", [None, (10, 10, 20, 20)])
@pytest.mark.parametrize("shape_order", ["HWC", "CHW"])
def test_image(img_shape, from_space_type, to_space_type, resize, trimming, shape_order, true_base_shape):
    pytest.importorskip("cv2")

    img = np.ones(img_shape).astype(np.uint8)
    img = helper.image_processor(img, from_space_type, to_space_type, resize, trimming, shape_order)

    # trimming -> resize
    assert len(true_base_shape) == 2
    true_shape = true_base_shape
    if resize is not None:
        assert len(resize) == 2
        true_shape = (resize[1], resize[0])
    elif trimming is not None:
        true_shape = (10, 10)

    if to_space_type == SpaceTypes.GRAY_3ch:
        true_shape = true_shape + (1,)
        if shape_order == "CHW":
            true_shape = (true_shape[2], true_shape[0], true_shape[1])
    elif to_space_type == SpaceTypes.COLOR:
        true_shape = true_shape + (3,)
        if shape_order == "CHW":
            true_shape = (true_shape[2], true_shape[0], true_shape[1])

    assert img.shape == true_shape
