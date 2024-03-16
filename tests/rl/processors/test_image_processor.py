from typing import cast

import numpy as np
import pytest

import srl
import srl.envs.grid
from srl.base.define import EnvTypes
from srl.base.rl.config import DummyRLConfig
from srl.base.spaces.box import BoxSpace
from srl.rl.processors.image_processor import ImageProcessor
from srl.test.processor import TestProcessor

image_w = 32
image_h = 64
image_resize = (84, 84)
enable_norm = True

test_pattens = (
    (EnvTypes.GRAY_2ch, (image_w, image_h), EnvTypes.GRAY_2ch, (84, 84), True),
    (EnvTypes.GRAY_2ch, (image_w, image_h), EnvTypes.GRAY_3ch, (84, 84, 1), True),
    (EnvTypes.GRAY_2ch, (image_w, image_h), EnvTypes.COLOR, (84, 84, 3), False),
    (EnvTypes.GRAY_3ch, (image_w, image_h, 1), EnvTypes.GRAY_2ch, (84, 84), True),
    (EnvTypes.GRAY_3ch, (image_w, image_h, 1), EnvTypes.GRAY_3ch, (84, 84, 1), True),
    (EnvTypes.GRAY_3ch, (image_w, image_h, 1), EnvTypes.COLOR, (84, 84, 3), False),
    (EnvTypes.COLOR, (image_w, image_h, 3), EnvTypes.GRAY_2ch, (84, 84), True),
    (EnvTypes.COLOR, (image_w, image_h, 3), EnvTypes.GRAY_3ch, (84, 84, 1), True),
    (EnvTypes.COLOR, (image_w, image_h, 3), EnvTypes.COLOR, (84, 84, 3), True),
)


@pytest.mark.parametrize("env_img_type, env_img_shape, img_type, true_shape, check_val", test_pattens)
def test_image(env_img_type, env_img_shape, img_type, true_shape, check_val):
    pytest.importorskip("cv2")

    processor = ImageProcessor(
        image_type=img_type,
        resize=image_resize,
        enable_norm=enable_norm,
    )
    space = BoxSpace(low=0, high=255, shape=env_img_shape)
    env = srl.make_env("Grid")

    # change info
    new_space, new_type = processor.preprocess_observation_space(
        space,
        env_img_type,
        env,
        DummyRLConfig(),
    )
    assert new_type == img_type
    assert isinstance(new_space, BoxSpace)
    new_space = cast(BoxSpace, new_space)
    assert new_space.shape == true_shape
    np.testing.assert_array_equal(new_space.low, np.full(true_shape, 0))
    np.testing.assert_array_equal(new_space.high, np.full(true_shape, 1))

    # decode
    image = np.ones(env_img_shape).astype(np.float32)  # image
    true_state = np.ones(true_shape).astype(np.float32) / 255
    new_obs = cast(np.ndarray, processor.preprocess_observation(image, env))
    assert true_state.shape == new_obs.shape
    if check_val:
        np.testing.assert_array_equal(true_state, new_obs)


def test_image_atari():
    pytest.importorskip("cv2")
    pytest.importorskip("ale_py")

    tester = TestProcessor()
    processor = ImageProcessor(
        image_type=EnvTypes.GRAY_2ch,
        resize=(84, 84),
        enable_norm=True,
    )
    env_name = "ALE/Tetris-v5"
    in_image = np.ones((210, 160, 3)).astype(np.float32)
    out_image = np.ones((84, 84)).astype(np.float32) / 255

    tester.run(processor, env_name)
    tester.preprocess_observation_space(
        processor,
        env_name,
        EnvTypes.GRAY_2ch,
        BoxSpace((84, 84), 0, 1),
    )
    tester.preprocess_observation(
        processor,
        env_name,
        in_observation=in_image,
        out_observation=out_image,
    )


def test_trimming():
    pytest.importorskip("cv2")

    space = BoxSpace(low=0, high=255, shape=(210, 160, 3))

    processor = ImageProcessor(
        image_type=EnvTypes.GRAY_2ch,
        trimming=(10, 10, 20, 20),
    )
    env = srl.make_env("Grid")

    # change info
    new_space, new_type = processor.preprocess_observation_space(
        space,
        EnvTypes.COLOR,
        env,
        DummyRLConfig(),
    )
    assert new_type == EnvTypes.GRAY_2ch
    assert isinstance(new_space, BoxSpace)
    new_space = cast(BoxSpace, new_space)
    assert new_space.shape == (10, 10)
    np.testing.assert_array_equal(new_space.low, np.full((10, 10), 0))
    np.testing.assert_array_equal(new_space.high, np.full((10, 10), 255))

    # decode
    image = np.ones((210, 160, 3)).astype(np.uint8)  # image
    true_state = np.ones((10, 10)).astype(np.float32) / 255
    new_obs = cast(np.ndarray, processor.preprocess_observation(image, env))
    assert true_state.shape == new_obs.shape
