import unittest

import numpy as np

from srl.rl.models.r2d3_image_block_config import R2D3ImageBlockConfig
from srl.utils.common import is_package_installed


class Test(unittest.TestCase):
    @unittest.skipUnless(is_package_installed("tensorflow"), "no module")
    def test_call_tf(self):
        config = R2D3ImageBlockConfig()
        batch_size = 16
        x = np.ones((batch_size, 96, 72, 3), dtype=np.float32)

        block = config.create_block_tf()
        y = block(x).numpy()

        self.assertTrue(y.shape == (batch_size, 12, 9, 32))

    @unittest.skipUnless(is_package_installed("torch"), "no module")
    def test_call_torch(self):
        import torch

        config = R2D3ImageBlockConfig()
        batch_size = 16
        x = np.ones((batch_size, 3, 96, 72), dtype=np.float32)

        x = torch.tensor(x)
        block = config.create_block_torch(x.shape[1:])
        y = block(x)
        y = y.detach().numpy()

        self.assertTrue(y.shape == (batch_size, 32, 12, 9))
        self.assertTrue(block.out_shape == (32, 12, 9))

    @unittest.skipUnless(is_package_installed("tensorflow"), "no module")
    def test_call_lstm_tf(self):
        config = R2D3ImageBlockConfig(enable_time_distributed_layer=True)
        batch_size = 16
        seq_len = 7
        x = np.ones((batch_size, seq_len, 96, 72, 3), dtype=np.float32)

        block = config.create_block_tf()
        y = block(x).numpy()

        self.assertTrue(y.shape == (batch_size, seq_len, 12, 9, 32))

    @unittest.skipUnless(is_package_installed("torch"), "no module")
    def test_call_lstm_torch(self):
        import torch

        config = R2D3ImageBlockConfig(enable_time_distributed_layer=True)
        batch_size = 16
        seq_len = 7
        x = np.ones((batch_size, seq_len, 3, 96, 72), dtype=np.float32)

        x = torch.tensor(x)
        block = config.create_block_torch(x.shape[1:])
        y = block(x)
        y = y.detach().numpy()

        self.assertTrue(y.shape == (batch_size, seq_len, 32, 12, 9))
        self.assertTrue(block.out_shape == (32, 12, 9))


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_call_lstm_tf", verbosity=2)
