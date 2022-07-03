import unittest

import numpy as np
from srl.rl.models.alphazero_image_block import AlphaZeroImageBlock
from tensorflow.keras import layers as kl


class Test(unittest.TestCase):
    def test_call(self):
        block = AlphaZeroImageBlock()
        batch_size = 16

        x = np.ones((batch_size, 64, 75, 19))
        out_x = block(x)

        self.assertTrue(out_x.shape == (batch_size, 64, 75, 256))

    def test_call_lstm(self):
        block = AlphaZeroImageBlock()
        batch_size = 16

        x = np.ones((batch_size, 1, 96, 72, 3))
        out_x = kl.TimeDistributed(block)(x)

        self.assertTrue(out_x.shape == (batch_size, 1, 96, 72, 256))


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_call_lstm", verbosity=2)
