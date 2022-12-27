import unittest

import numpy as np

from srl.base.env.spaces import ArrayDiscreteSpace
from tests.base.env.space_test import SpaceTest


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.space = ArrayDiscreteSpace(3, 0, [2, 5, 3])
        self.assertTrue(self.space.size, 3)
        np.testing.assert_array_equal(self.space.low, [0, 0, 0])
        np.testing.assert_array_equal(self.space.high, [2, 5, 3])

        self.tester = SpaceTest(self, self.space)

    def _check_action(self, decode_action, size, true_action):
        self.assertTrue(isinstance(decode_action, list))
        self.assertTrue(len(decode_action) == size)
        for a in decode_action:
            self.assertTrue(isinstance(a, int))
        if true_action is not None:
            np.testing.assert_array_equal(decode_action, true_action)

    def test_space(self):
        # sample
        for _ in range(100):
            action = self.space.sample()
            self._check_action(action, 3, None)
            self.assertTrue(0 <= action[0] <= 2)
            self.assertTrue(0 <= action[1] <= 5)
            self.assertTrue(0 <= action[2] <= 3)

        # action discrete
        decode_action = self.tester.check_action_discrete(
            true_n=(2 + 1) * (5 + 1) * (3 + 1),
            action=1,
        )
        self._check_action(decode_action, 3, [0, 0, 1])
        self.tester.check_action_encode([0, 0, 1], 1)

        # action_continuous
        decode_action = self.tester.check_action_continuous(
            true_n=3,
            true_low=[0, 0, 0],
            true_high=[2, 5, 3],
            action=[0.1, 0.6, 0.9],
        )
        self._check_action(decode_action, 3, [0, 1, 1])

        # observation discrete
        self.tester.check_observation_discrete(
            true_shape=(3,),
            state=[0, 0, 1],
            encode_state=[0, 0, 1],
        )

        # observation continuous
        self.tester.check_observation_continuous(
            true_shape=(3,),
            state=[0, 0, 1],
            encode_state=[0, 0, 1],
        )

        # eq
        self.assertTrue(self.space == ArrayDiscreteSpace(3, 0, [2, 5, 3]))
        self.assertTrue(self.space != ArrayDiscreteSpace(3, 0, [3, 5, 3]))

    def test_convert(self):
        val = self.space.convert(1.2)
        np.testing.assert_array_equal([1], val)

        val = self.space.convert([1.2, 0.9])
        np.testing.assert_array_equal([1, 1], val)

        val = self.space.convert((9, 10, True))
        np.testing.assert_array_equal([9, 10, 1], val)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_space", verbosity=2)
