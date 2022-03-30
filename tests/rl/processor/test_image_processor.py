import unittest

import gym
import gym.spaces
import numpy as np
from srl.base.define import EnvObservationType, RLObservationType
from srl.rl.processor.image_processor import ImageProcessor


class Test(unittest.TestCase):
    def test_image(self):
        image_w = 32
        image_h = 64

        test_pattens = (
            (EnvObservationType.COLOR, (image_w, image_h, 3), (84, 84)),
            (EnvObservationType.GRAY_3ch, (image_w, image_h, 1), (84, 84)),
        )
        for env_type, img_shape, image_resize in test_pattens:
            with self.subTest(f"COLOR {env_type} {img_shape} {image_resize}"):
                processor = ImageProcessor(
                    change_gray=True,
                    resize=image_resize,
                    enable_norm=True,
                )
                space = gym.spaces.Box(low=0, high=255, shape=(image_w, image_h, 3))

                # change info
                new_space, new_type = processor.change_observation_info(space, env_type, RLObservationType.UNKOWN)
                self.assertTrue(new_type == EnvObservationType.GRAY_2ch)
                self.assertTrue(new_space.shape == image_resize)
                np.testing.assert_array_equal(new_space.low, np.full(image_resize, 0))
                np.testing.assert_array_equal(new_space.high, np.full(image_resize, 1))

                # decode
                image = np.ones(img_shape).astype(np.float32)  # image
                true_state = np.ones(image_resize).astype(np.float32) / 255
                new_obs = processor.observation_encode(image)
                np.testing.assert_array_equal(true_state, new_obs)


if __name__ == "__main__":
    name = "test_image"
    unittest.main(module=__name__, defaultTest="Test." + name, verbosity=2)
