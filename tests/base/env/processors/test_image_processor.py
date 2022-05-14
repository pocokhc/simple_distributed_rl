import unittest
from typing import cast

import numpy as np
from srl.base.define import EnvObservationType, RLObservationType
from srl.base.env.processors import ImageProcessor
from srl.base.env.spaces.box import BoxSpace


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
                    gray=True,
                    resize=image_resize,
                    enable_norm=True,
                )
                space = BoxSpace(low=0, high=255, shape=img_shape)

                # change info
                new_space, new_type = processor.change_observation_info(space, env_type, RLObservationType.ANY, None)
                self.assertTrue(new_type == EnvObservationType.GRAY_2ch)
                self.assertTrue(isinstance(new_space, BoxSpace))
                new_space = cast(BoxSpace, new_space)
                self.assertTrue(new_space.shape == image_resize)
                np.testing.assert_array_equal(new_space.low, np.full(image_resize, 0))
                np.testing.assert_array_equal(new_space.high, np.full(image_resize, 1))

                # decode
                image = np.ones(img_shape).astype(np.float32)  # image
                true_state = np.ones(image_resize).astype(np.float32) / 255
                new_obs = processor.process_observation(image, None)
                np.testing.assert_array_equal(true_state, new_obs)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_image", verbosity=2)
