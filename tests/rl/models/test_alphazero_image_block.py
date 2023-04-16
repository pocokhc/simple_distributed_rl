import unittest

import numpy as np

from srl.rl.models.alphazero_image_block_config import AlphaZeroImageBlockConfig
from srl.utils.common import is_package_installed


class Test(unittest.TestCase):
    @unittest.skipUnless(is_package_installed("tensorflow"), "no module")
    def test_call_tf(self):
        config = AlphaZeroImageBlockConfig()
        batch_size = 16
        x = np.ones((batch_size, 96, 72, 3), dtype=np.float32)

        block = config.create_block_tf()
        y = block(x).numpy()

        self.assertTrue(y.shape == (batch_size, 96, 72, 256))

    @unittest.skipUnless(is_package_installed("torch"), "no module")
    def test_call_torch(self):
        import torch

        config = AlphaZeroImageBlockConfig()
        batch_size = 16
        x = np.ones((batch_size, 3, 96, 72), dtype=np.float32)

        x = torch.tensor(x)
        block = config.create_block_torch(x.shape[1:])
        y = block(x)
        y = y.detach().numpy()

        self.assertTrue(y.shape == (batch_size, 256, 96, 72))
        self.assertTrue(block.out_shape == (256, 96, 72))


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_call_torch", verbosity=2)
