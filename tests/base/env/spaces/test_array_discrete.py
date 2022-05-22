import unittest

import numpy as np
from srl.base.env.spaces import ArrayDiscreteSpace
from tests.base.env.space_test import SpaceTest


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.space = ArrayDiscreteSpace([2, 5, 3])
        np.testing.assert_array_equal(self.space.nvec, [2, 5, 3])

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
            self.assertTrue(0 <= action[0] <= 1)
            self.assertTrue(0 <= action[1] <= 4)
            self.assertTrue(0 <= action[2] <= 2)

        # action discrete
        decode_action = self.tester.check_action_discrete(
            true_n=2 * 5 * 3,
            action=1,
        )
        self._check_action(decode_action, 3, [0, 0, 1])
        self.tester.check_action_encode([0, 0, 1], 1)

        # action_continuous
        decode_action = self.tester.check_action_continuous(
            true_n=3,
            true_low=[0, 0, 0],
            true_high=[1, 4, 2],
            action=[0.1, 0.6, 0.9],
        )
        self._check_action(decode_action, 3, [0, 1, 1])

        # observation discrete
        self.tester.check_observation_discrete(
            true_shape=(3,),
            true_low=[0, 0, 0],
            true_high=[1, 4, 2],
            state=[0, 0, 1],
            encode_state=[0, 0, 1],
        )

        # observation continuous
        self.tester.check_observation_continuous(
            true_shape=(3,),
            true_low=[0, 0, 0],
            true_high=[1, 4, 2],
            state=[0, 0, 1],
            encode_state=[0, 0, 1],
        )


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_space", verbosity=2)
