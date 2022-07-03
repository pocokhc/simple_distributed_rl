import unittest

import numpy as np
from srl.rl.models.mlp_block import MLPBlock


class Test(unittest.TestCase):
    def test_call(self):
        block = MLPBlock(hidden_layer_sizes=(128, 512))
        batch_size = 16

        x = np.ones((batch_size, 8))
        out_x = block(x)

        self.assertTrue(out_x.shape == (batch_size, 512))


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_call", verbosity=2)
