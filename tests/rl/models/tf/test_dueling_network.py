import unittest

import numpy as np

from srl.utils.common import is_package_installed

try:
    from srl.rl.models.tf.dueling_network import DuelingNetworkBlock
except ModuleNotFoundError:
    pass


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test(unittest.TestCase):
    def test_call(self):
        action_num = 5
        dense_units = 32
        batch_size = 16

        block = DuelingNetworkBlock(action_num, dense_units)

        x = np.ones((batch_size, 128))
        out_x = block(x)

        self.assertTrue(out_x.shape == (batch_size, action_num))


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_call", verbosity=2)
