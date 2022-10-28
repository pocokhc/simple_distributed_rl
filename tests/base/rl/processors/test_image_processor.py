import unittest
from typing import cast

import numpy as np
from srl.base.define import EnvObservationType, RLObservationType
from srl.base.env.spaces.box import BoxSpace
from srl.test.processor import TestProcessor
from srl.utils.common import is_package_installed

try:
    from srl.base.rl.processors import ImageProcessor
except ModuleNotFoundError:
    pass


@unittest.skipUnless(is_package_installed("cv2"), "no module")
class Test(unittest.TestCase):
    def test_image(self):
        image_w = 32
        image_h = 64
        image_resize = (84, 84)
        enable_norm = True

        test_pattens = (
            (EnvObservationType.GRAY_2ch, (image_w, image_h), EnvObservationType.GRAY_2ch, (84, 84), True),
            (EnvObservationType.GRAY_2ch, (image_w, image_h), EnvObservationType.GRAY_3ch, (84, 84, 1), True),
            (EnvObservationType.GRAY_2ch, (image_w, image_h), EnvObservationType.COLOR, (84, 84, 3), False),
            (EnvObservationType.GRAY_3ch, (image_w, image_h, 1), EnvObservationType.GRAY_2ch, (84, 84), True),
            (EnvObservationType.GRAY_3ch, (image_w, image_h, 1), EnvObservationType.GRAY_3ch, (84, 84, 1), True),
            (EnvObservationType.GRAY_3ch, (image_w, image_h, 1), EnvObservationType.COLOR, (84, 84, 3), False),
            (EnvObservationType.COLOR, (image_w, image_h, 3), EnvObservationType.GRAY_2ch, (84, 84), True),
            (EnvObservationType.COLOR, (image_w, image_h, 3), EnvObservationType.GRAY_3ch, (84, 84, 1), True),
            (EnvObservationType.COLOR, (image_w, image_h, 3), EnvObservationType.COLOR, (84, 84, 3), True),
        )
        for env_img_type, env_img_shape, img_type, true_shape, check_val in test_pattens:
            with self.subTest(f"{env_img_type} {env_img_shape} {img_type} {image_resize}"):
                processor = ImageProcessor(
                    image_type=img_type,
                    resize=image_resize,
                    enable_norm=enable_norm,
                )
                space = BoxSpace(low=0, high=255, shape=env_img_shape)

                # change info
                new_space, new_type = processor.change_observation_info(
                    space, env_img_type, RLObservationType.ANY, None
                )
                self.assertTrue(new_type == img_type)
                self.assertTrue(isinstance(new_space, BoxSpace))
                new_space = cast(BoxSpace, new_space)
                self.assertTrue(new_space.shape == true_shape)
                np.testing.assert_array_equal(new_space.low, np.full(true_shape, 0))
                np.testing.assert_array_equal(new_space.high, np.full(true_shape, 1))

                # decode
                image = np.ones(env_img_shape).astype(np.float32)  # image
                true_state = np.ones(true_shape).astype(np.float32) / 255
                new_obs = processor.process_observation(image, None)
                self.assertTrue(true_state.shape == new_obs.shape)
                if check_val:
                    np.testing.assert_array_equal(true_state, new_obs)

    @unittest.skipUnless(is_package_installed("ale_py"), "no module")
    def test_image_atari(self):
        tester = TestProcessor()
        processor = ImageProcessor(
            image_type=EnvObservationType.GRAY_2ch,
            resize=(84, 84),
            enable_norm=True,
        )
        env_name = "ALE/Tetris-v5"
        in_image = np.ones((210, 160, 3)).astype(np.float32)
        out_image = np.ones((84, 84)).astype(np.float32) / 255

        tester.run(processor, env_name)
        tester.change_observation_info(
            processor,
            env_name,
            EnvObservationType.GRAY_2ch,
            BoxSpace((84, 84), 0, 1),
        )
        tester.observation_decode(
            processor,
            env_name,
            in_observation=in_image,
            out_observation=out_image,
        )

    def test_trimming(self):
        space = BoxSpace(low=0, high=255, shape=(210, 160, 3))

        processor = ImageProcessor(
            image_type=EnvObservationType.GRAY_2ch,
            trimming=(10, 10, 20, 20),
        )

        # change info
        new_space, new_type = processor.change_observation_info(
            space, EnvObservationType.COLOR, RLObservationType.ANY, None
        )
        self.assertTrue(new_type == EnvObservationType.GRAY_2ch)
        self.assertTrue(isinstance(new_space, BoxSpace))
        new_space = cast(BoxSpace, new_space)
        self.assertTrue(new_space.shape == (10, 10))
        np.testing.assert_array_equal(new_space.low, np.full((10, 10), 0))
        np.testing.assert_array_equal(new_space.high, np.full((10, 10), 255))

        # decode
        image = np.ones((210, 160, 3)).astype(np.uint8)  # image
        true_state = np.ones((10, 10)).astype(np.float32) / 255
        new_obs = processor.process_observation(image, None)
        self.assertTrue(true_state.shape == new_obs.shape)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_trimming", verbosity=2)
