from typing import cast

import numpy as np
import pytest

import srl
from srl.base.define import SpaceTypes
from srl.base.spaces.box import BoxSpace
from srl.rl.processors.image_processor import ImageProcessor

image_w = 32
image_h = 64
image_resize = (84, 84)

test_pattens = (
    (SpaceTypes.GRAY_HW, (image_w, image_h), SpaceTypes.GRAY_HW, (84, 84), True),
    (SpaceTypes.GRAY_HW, (image_w, image_h), SpaceTypes.GRAY_HW1, (84, 84, 1), True),
    (SpaceTypes.GRAY_HW, (image_w, image_h), SpaceTypes.RGB, (84, 84, 3), False),
    (SpaceTypes.GRAY_HW1, (image_w, image_h, 1), SpaceTypes.GRAY_HW, (84, 84), True),
    (SpaceTypes.GRAY_HW1, (image_w, image_h, 1), SpaceTypes.GRAY_HW1, (84, 84, 1), True),
    (SpaceTypes.GRAY_HW1, (image_w, image_h, 1), SpaceTypes.RGB, (84, 84, 3), False),
    (SpaceTypes.RGB, (image_w, image_h, 3), SpaceTypes.GRAY_HW, (84, 84), True),
    (SpaceTypes.RGB, (image_w, image_h, 3), SpaceTypes.GRAY_HW1, (84, 84, 1), True),
    (SpaceTypes.RGB, (image_w, image_h, 3), SpaceTypes.RGB, (84, 84, 3), True),
)


@pytest.mark.parametrize("env_img_type, env_img_shape, img_type, true_shape, check_val", test_pattens)
@pytest.mark.parametrize("normalize_type", ["", "0to1", "-1to1"])
def test_image(env_img_type, env_img_shape, img_type, true_shape, check_val, normalize_type):
    pytest.importorskip("cv2")

    processor = ImageProcessor(
        image_type=img_type,
        resize=image_resize,
        normalize_type=normalize_type,
    )
    space = BoxSpace(
        low=0,
        high=255,
        shape=env_img_shape,
        dtype=np.uint8,
        stype=env_img_type,
    )

    # --- change space
    new_space = processor.remap_observation_space(space)
    assert new_space is not None
    assert new_space.stype == img_type
    if normalize_type == "0to1" or normalize_type == "-1to1":
        assert new_space.dtype == np.float32
    else:
        assert new_space.dtype == np.uint8
    assert isinstance(new_space, BoxSpace)
    new_space = cast(BoxSpace, new_space)
    assert new_space.shape == true_shape
    if normalize_type == "0to1":
        np.testing.assert_array_equal(new_space.low, np.full(true_shape, 0))
        np.testing.assert_array_equal(new_space.high, np.full(true_shape, 1))
    elif normalize_type == "-1to1":
        np.testing.assert_array_equal(new_space.low, np.full(true_shape, -1))
        np.testing.assert_array_equal(new_space.high, np.full(true_shape, 1))
    else:
        np.testing.assert_array_equal(new_space.low, np.full(true_shape, 0))
        np.testing.assert_array_equal(new_space.high, np.full(true_shape, 255))

    # --- decode
    image = np.ones(env_img_shape).astype(np.uint8)  # image
    if normalize_type == "0to1":
        true_state = np.ones(true_shape).astype(np.float32) / 255
    elif normalize_type == "-1to1":
        true_state = np.ones(true_shape).astype(np.float32) / 255 * 2 - 1
    else:
        true_state = np.ones(true_shape).astype(np.uint8)
    new_obs = cast(np.ndarray, processor.remap_observation(image, space, new_space))
    assert true_state.shape == new_obs.shape
    if check_val:
        np.testing.assert_array_equal(true_state, new_obs)


def test_image_atari():
    pytest.importorskip("cv2")
    pytest.importorskip("ale_py")

    processor = ImageProcessor(
        image_type=SpaceTypes.GRAY_HW,
        resize=(84, 84),
        normalize_type="0to1",
    )
    env = srl.make_env("ALE/Tetris-v5")
    env.setup()
    env.reset()
    in_image = np.ones((210, 160, 3))
    out_image = np.ones((84, 84)).astype(np.float32) / 255

    assert in_image.shape == env.state.shape

    # --- space
    new_space = processor.remap_observation_space(env.observation_space)
    assert new_space is not None
    assert new_space.stype == SpaceTypes.GRAY_HW
    assert new_space.dtype == np.float32
    assert isinstance(new_space, BoxSpace)
    assert new_space.shape == out_image.shape

    # --- decode
    new_state = processor.remap_observation(env.state, env.observation_space, new_space)
    assert isinstance(new_state, np.ndarray)
    assert new_state.shape == out_image.shape


def test_trimming():
    pytest.importorskip("cv2")

    space = BoxSpace(low=0, high=255, shape=(210, 160, 3), stype=SpaceTypes.RGB)

    processor = ImageProcessor(
        image_type=SpaceTypes.GRAY_HW,
        trimming=(10, 10, 20, 20),
    )

    # --- space
    new_space = processor.remap_observation_space(space)
    assert new_space is not None
    assert new_space.stype == SpaceTypes.GRAY_HW
    assert isinstance(new_space, BoxSpace)
    new_space = cast(BoxSpace, new_space)
    assert new_space.shape == (10, 10)
    np.testing.assert_array_equal(new_space.low, np.full((10, 10), 0))
    np.testing.assert_array_equal(new_space.high, np.full((10, 10), 255))

    # --- decode
    image = np.ones((210, 160, 3)).astype(np.uint8)  # image
    true_state = np.ones((10, 10)).astype(np.float32) / 255
    new_state = processor.remap_observation(image, space, new_space)
    assert isinstance(new_state, np.ndarray)
    assert true_state.shape == new_state.shape
