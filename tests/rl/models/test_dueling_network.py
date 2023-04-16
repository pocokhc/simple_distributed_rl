import unittest

import numpy as np

from srl.utils.common import is_package_installed


class Test(unittest.TestCase):
    @unittest.skipUnless(is_package_installed("tensorflow"), "no module")
    def test_call_tf(self):
        from srl.rl.models.tf.dueling_network import DuelingNetworkBlock

        action_num = 5
        dense_units = 32
        batch_size = 16

        block = DuelingNetworkBlock(action_num, dense_units)

        x = np.ones((batch_size, 128), dtype=np.float32)
        y = block(x).numpy()

        self.assertTrue(y.shape == (batch_size, action_num))

    @unittest.skipUnless(is_package_installed("torch"), "no module")
    def test_call_torch(self):
        import torch

        from srl.rl.models.torch_.dueling_network import DuelingNetworkBlock

        action_num = 5
        dense_units = 32
        batch_size = 16

        block = DuelingNetworkBlock(128, action_num, dense_units)

        x = np.ones((batch_size, 128), dtype=np.float32)
        y = block(torch.tensor(x))
        y = y.detach().numpy()

        self.assertTrue(y.shape == (batch_size, action_num))


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_call_torch", verbosity=2)
