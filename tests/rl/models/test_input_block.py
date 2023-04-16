import unittest

import numpy as np

from srl.base.define import EnvObservationType
from srl.utils.common import is_package_installed


class Test(unittest.TestCase):
    def call_block_tf(
        self,
        obs_shape,
        obs_type,
        x,
        enable_time_distributed_layer=False,
    ):
        from srl.rl.models.tf.input_block import InputBlock

        block = InputBlock(
            obs_shape,
            obs_type,
            enable_time_distributed_layer=enable_time_distributed_layer,
        )
        y = block(x).numpy()
        return y, block.use_image_layer, None

    def call_block_torch(
        self,
        obs_shape,
        obs_type,
        x,
        enable_time_distributed_layer=False,
    ):
        import torch

        from srl.rl.models.torch_.input_block import InputBlock

        block = InputBlock(
            obs_shape,
            obs_type,
            enable_time_distributed_layer=enable_time_distributed_layer,
        )
        y = block(torch.tensor(x))
        out_shape = block.out_shape

        if len(y.shape) == 4:
            # (batch, ch, h, w) -> (batch, h, w, ch)
            y = y.permute((0, 2, 3, 1))
            out_shape = (out_shape[1], out_shape[2], out_shape[0])
        if len(y.shape) == 5:
            # (batch, len, ch, h, w) -> (batch, len, h, w, ch)
            y = y.permute((0, 1, 3, 4, 2))
            out_shape = (out_shape[1], out_shape[2], out_shape[0])

        y = y.detach().numpy()
        return y, block.use_image_layer, out_shape

    @unittest.skipUnless(is_package_installed("tensorflow"), "no module")
    def test_window_0_tf(self):
        self._window_0(self.call_block_tf)

    @unittest.skipUnless(is_package_installed("torch"), "no module")
    def test_window_0_torch(self):
        self._window_0(self.call_block_torch)

    def _window_0(self, call_block):
        batch_size = 8

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
                x = np.ones((batch_size,) + obs_shape, dtype=np.float32)
                y, use_image_layer, out_shape = call_block(obs_shape, obs_type, x)
                self.assertTrue(use_image_layer == true_image)
                self.assertTrue(y.shape == (batch_size,) + true_shape)
                if out_shape is not None:
                    self.assertTrue(out_shape == true_shape)

    @unittest.skipUnless(is_package_installed("tensorflow"), "no module")
    def test_window_10_tf(self):
        self._window_10(self.call_block_tf)

    @unittest.skipUnless(is_package_installed("torch"), "no module")
    def test_window_10_torch(self):
        self._window_10(self.call_block_torch)

    def _window_10(self, call_block):
        batch_size = 8

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
                x = np.ones((batch_size,) + obs_shape, dtype=np.float32)
                if is_throw:
                    with self.assertRaises(ValueError):
                        call_block(obs_shape, obs_type, x)
                else:
                    y, use_image_layer, out_shape = call_block(obs_shape, obs_type, x)
                    self.assertTrue(use_image_layer == true_image)
                    self.assertTrue(y.shape == (batch_size,) + true_shape)
                    if out_shape is not None:
                        self.assertTrue(out_shape == true_shape)

    @unittest.skipUnless(is_package_installed("tensorflow"), "no module")
    def test_time_layer_tf(self):
        self._time_layer(self.call_block_tf)

    @unittest.skipUnless(is_package_installed("torch"), "no module")
    def test_time_layer_torch(self):
        self._time_layer(self.call_block_torch)

    def _time_layer(self, call_block):
        batch_size = 7
        seq_len = 3

        for obs_shape, obs_type, true_shape, true_image in [
            # ((2, 4, 8), EnvObservationType.UNKNOWN, (2 * 4 * 8,), False),
            # ((2, 4, 8), EnvObservationType.DISCRETE, (2 * 4 * 8,), False),
            # ((2, 4, 8), EnvObservationType.CONTINUOUS, (2 * 4 * 8,), False),
            # ((4, 8), EnvObservationType.GRAY_2ch, (4, 8, 1), True),
            # ((4, 8, 1), EnvObservationType.GRAY_3ch, (4, 8, 1), True),
            # ((4, 8, 3), EnvObservationType.COLOR, (4, 8, 3), True),
            # ((4, 8), EnvObservationType.SHAPE2, (4, 8, 1), True),
            ((10, 4, 8), EnvObservationType.SHAPE3, (4, 8, 10), True),
        ]:
            with self.subTest((obs_shape, obs_type, true_shape, true_image)):
                x = np.ones((batch_size, seq_len) + obs_shape, dtype=np.float32)
                y, use_image_layer, out_shape = call_block(obs_shape, obs_type, x, enable_time_distributed_layer=True)
                self.assertTrue(use_image_layer == true_image)
                self.assertTrue(y.shape == (batch_size, seq_len) + true_shape)
                if out_shape is not None:
                    self.assertTrue(out_shape == true_shape)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_time_layer_torch", verbosity=2)
