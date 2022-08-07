import unittest

from srl.base.define import EnvObservationType
from srl.rl.models.input_layer import create_input_layer, create_input_layer_stateful_lstm


class Test(unittest.TestCase):
    def test_no_window(self):
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
                in_layer, out_layer, use_image_head = create_input_layer(obs_shape, obs_type)
                self.assertTrue(use_image_head == true_image)
                self.assertTrue(in_layer.shape == (None,) + obs_shape)
                self.assertTrue(out_layer.shape == (None,) + true_shape)

    def test_window_10(self):
        for obs_shape, obs_type, true_shape, true_image, is_throw in [
            ((10, 2, 4, 8), EnvObservationType.UNKNOWN, (10 * 2 * 4 * 8,), False, False),
            ((10, 2, 4, 8), EnvObservationType.DISCRETE, (10 * 2 * 4 * 8,), False, False),
            ((10, 2, 4, 8), EnvObservationType.CONTINUOUS, (10 * 2 * 4 * 8,), False, False),
            ((10, 4, 8), EnvObservationType.GRAY_2ch, (4, 8, 10), True, False),
            ((10, 4, 8, 1), EnvObservationType.GRAY_3ch, (4, 8, 10), True, False),
            ((10, 4, 8, 3), EnvObservationType.COLOR, None, None, True),
            ((10, 4, 8), EnvObservationType.SHAPE2, (4, 8, 10), True, False),
            ((10, 10, 4, 8), EnvObservationType.SHAPE3, None, None, True),
        ]:
            with self.subTest((obs_shape, obs_type, true_shape, true_image)):
                if is_throw:
                    with self.assertRaises(ValueError):
                        create_input_layer(obs_shape, obs_type)
                else:
                    in_layer, out_layer, use_image_head = create_input_layer(obs_shape, obs_type)
                    self.assertTrue(use_image_head == true_image)
                    self.assertTrue(in_layer.shape == (None,) + obs_shape)
                    self.assertTrue(out_layer.shape == (None,) + true_shape)

    def test_stateful_lstm_no_window(self):
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
                in_layer, out_layer, use_image_head = create_input_layer_stateful_lstm(batch_size, obs_shape, obs_type)
                self.assertTrue(use_image_head == true_image)
                self.assertTrue(in_layer.shape == (batch_size, 1) + obs_shape)
                self.assertTrue(out_layer.shape == (batch_size, 1) + true_shape)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_window_10", verbosity=2)
