import unittest

import numpy as np

from srl.rl.models.dqn_image_block_config import DQNImageBlockConfig
from srl.utils.common import is_package_installed


class Test(unittest.TestCase):
    @unittest.skipUnless(is_package_installed("tensorflow"), "no module")
    def test_call_tf(self):
        config = DQNImageBlockConfig()
        batch_size = 16
        x = np.ones((batch_size, 64, 75, 19), dtype=np.float32)

        block = config.create_block_tf()
        y = block(x).numpy()

        self.assertTrue(y.shape == (batch_size, 8, 10, 64))

    @unittest.skipUnless(is_package_installed("torch"), "no module")
    def test_call_torch(self):
        import torch

        config = DQNImageBlockConfig()
        batch_size = 16
        x = np.ones((batch_size, 19, 64, 75), dtype=np.float32)

        x = torch.tensor(x)
        block = config.create_block_torch(x.shape[1:])
        y = block(x)
        y = y.detach().numpy()

        self.assertTrue(y.shape == (batch_size, 64, 9, 10))
        self.assertTrue(block.out_shape == (64, 9, 10))

    @unittest.skipUnless(is_package_installed("tensorflow"), "no module")
    def test_call_lstm_tf(self):
        config = DQNImageBlockConfig(enable_time_distributed_layer=True)
        batch_size = 16
        seq_len = 7
        x = np.ones((batch_size, seq_len, 64, 75, 19), dtype=np.float32)

        block = config.create_block_tf()
        y = block(x).numpy()

        self.assertTrue(y.shape == (batch_size, seq_len, 8, 10, 64))

    @unittest.skipUnless(is_package_installed("torch"), "no module")
    def test_call_lstm_torch(self):
        import torch

        config = DQNImageBlockConfig(enable_time_distributed_layer=True)
        batch_size = 16
        seq_len = 7
        x = np.ones((batch_size, seq_len, 19, 64, 75), dtype=np.float32)

        x = torch.tensor(x)
        block = config.create_block_torch(x.shape[1:])
        y = block(x)
        y = y.detach().numpy()

        self.assertTrue(y.shape == (batch_size, seq_len, 64, 9, 10))
        self.assertTrue(block.out_shape == (64, 9, 10))


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_call_lstm_torch", verbosity=2)
