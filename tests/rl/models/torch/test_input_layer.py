import unittest

import numpy as np

from srl.base.define import EnvObservationType
from srl.utils.common import is_package_installed

try:
    import torch

    from srl.rl.models.torch.input_layer import InputLayer
except ModuleNotFoundError:
    pass


@unittest.skipUnless(is_package_installed("torch"), "no module")
class Test(unittest.TestCase):
    def test_no_window(self):
        batch_size = 16
        for obs_shape, obs_type, true_shape, true_image in [
            ((2, 4, 8), EnvObservationType.UNKNOWN, (2 * 4 * 8,), False),
            ((2, 4, 8), EnvObservationType.DISCRETE, (2 * 4 * 8,), False),
            ((2, 4, 8), EnvObservationType.CONTINUOUS, (2 * 4 * 8,), False),
            ((4, 8), EnvObservationType.GRAY_2ch, (4, 8, 1), True),
            ((4, 8, 1), EnvObservationType.GRAY_3ch, (4, 8, 1), True),
            ((4, 8, 3), EnvObservationType.COLOR, (4, 8, 3), True),
            ((4, 8), EnvObservationType.SHAPE2, (4, 8, 1), True),
            ((10, 4, 8), EnvObservationType.SHAPE3, (4, 8, 10), True),
        ]:
            with self.subTest((obs_shape, obs_type, true_shape, true_image)):
                layer = InputLayer(obs_shape, obs_type)
                self.assertTrue(layer.is_image_head() == true_image)

                x = torch.randn((batch_size,) + obs_shape)
                self.assertTrue(tuple(x.shape) == (batch_size,) + obs_shape)
                x = layer(x)
                self.assertTrue(tuple(x.shape) == (batch_size,) + true_shape)

    def test_window_10(self):
        batch_size = 16
        for obs_shape, obs_type, true_shape, true_image, is_throw in [
            ((10, 2, 4, 8), EnvObservationType.UNKNOWN, (10 * 2 * 4 * 8,), False, False),
            ((10, 2, 4, 8), EnvObservationType.DISCRETE, (10 * 2 * 4 * 8,), False, False),
            ((10, 2, 4, 8), EnvObservationType.CONTINUOUS, (10 * 2 * 4 * 8,), False, False),
            ((10, 4, 8), EnvObservationType.GRAY_2ch, (4, 8, 10), True, False),
            ((10, 4, 8, 1), EnvObservationType.GRAY_3ch, (4, 8, 10), True, False),
            ((10, 4, 8, 3), EnvObservationType.COLOR, None, True, True),
            ((10, 4, 8), EnvObservationType.SHAPE2, (4, 8, 10), True, False),
            ((10, 10, 4, 8), EnvObservationType.SHAPE3, None, True, True),
        ]:
            with self.subTest((obs_shape, obs_type, true_shape, true_image)):
                layer = InputLayer(obs_shape, obs_type)
                self.assertTrue(layer.is_image_head() == true_image)

                x = torch.randn((batch_size,) + obs_shape)
                self.assertTrue(tuple(x.shape) == (batch_size,) + obs_shape)
                if is_throw:
                    with self.assertRaises(ValueError):
                        x = layer(x)
                else:
                    x = layer(x)
                    self.assertTrue(tuple(x.shape) == (batch_size,) + true_shape)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_window_10", verbosity=2)
