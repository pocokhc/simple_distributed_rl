import unittest

import numpy as np
from srl.rl.models.dqn_image_block import DQNImageBlock
from tensorflow.keras import layers as kl


class Test(unittest.TestCase):
    def test_call(self):
        block = DQNImageBlock()
        batch_size = 16

        x = np.ones((batch_size, 64, 75, 19))
        out_x = block(x)

        self.assertTrue(out_x.shape == (batch_size, 8, 10, 64))

    def test_call_lstm(self):
        block = DQNImageBlock()
        batch_size = 16

        x = np.ones((batch_size, 1, 64, 75, 19))
        out_x = kl.TimeDistributed(block)(x)

        self.assertTrue(out_x.shape == (batch_size, 1, 8, 10, 64))


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_call", verbosity=2)
