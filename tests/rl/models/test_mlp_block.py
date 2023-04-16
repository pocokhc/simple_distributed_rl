import unittest

import numpy as np

from srl.rl.models.mlp_block_config import MLPBlockConfig
from srl.utils.common import is_package_installed


class Test(unittest.TestCase):
    @unittest.skipUnless(is_package_installed("tensorflow"), "no module")
    def test_call_tf(self):
        config = MLPBlockConfig((64, 32))
        batch_size = 16
        x = np.ones((batch_size, 256), dtype=np.float32)

        block = config.create_block_tf()
        y = block(x).numpy()

        self.assertTrue(y.shape == (batch_size, 32))

    @unittest.skipUnless(is_package_installed("torch"), "no module")
    def test_call_torch(self):
        import torch

        config = MLPBlockConfig((64, 32))
        batch_size = 16
        x = np.ones((batch_size, 256), dtype=np.float32)

        x = torch.tensor(x)
        block = config.create_block_torch(256)
        y = block(x)
        y = y.detach().numpy()

        self.assertTrue(y.shape == (batch_size, 32))
        self.assertTrue(block.out_shape == (32,))


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_call_tf", verbosity=2)
