import itertools
import unittest

import numpy as np
from srl.base.env.spaces import ArrayContinuousSpace
from tests.base.env.space_test import SpaceTest


class Test(unittest.TestCase):
    def _check_action(self, decode_action, size, true_action):
        self.assertTrue(isinstance(decode_action, list))
        self.assertTrue(len(decode_action) == size)
        for a in decode_action:
            self.assertTrue(isinstance(a, float))
        if true_action is not None:
            np.testing.assert_array_equal(decode_action, true_action)

    def test_space(self):
        self.space = ArrayContinuousSpace(3, -1, 3)
        self.tester = SpaceTest(self, self.space)

        # sample
        for _ in range(100):
            action = self.space.sample()
            self._check_action(action, 3, None)
            self.assertTrue(-1 <= action[0] <= 3)
            self.assertTrue(-1 <= action[1] <= 3)
            self.assertTrue(-1 <= action[2] <= 3)

        # action discrete
        self.space.set_action_division(5)
        decode_action = self.tester.check_action_discrete(
            true_n=5**3,
            action=3,
        )
        self._check_action(decode_action, 3, [-1, -1, 2])
        self.tester.check_action_encode(decode_action, 3)

        # action_continuous
        decode_action = self.tester.check_action_continuous(
            true_n=3,
            true_low=[-1] * 3,
            true_high=[3] * 3,
            action=[1.1, 0.1, 0.9],
        )
        self._check_action(decode_action, 3, [1.1, 0.1, 0.9])

        # observation discrete
        self.tester.check_observation_discrete(
            true_shape=(3,),
            state=[1.1, 0.1, 0.9],
            encode_state=[1, 0, 1],
        )

        # observation continuous
        self.tester.check_observation_continuous(
            true_shape=(3,),
            state=[1.1, 0.1, 0.9],
            encode_state=np.array([1.1, 0.1, 0.9], dtype=np.float32),
        )

    def test_inf(self):
        self.space = ArrayContinuousSpace(3)
        self.tester = SpaceTest(self, self.space)

        # sample
        for _ in range(100):
            action = self.space.sample()
            self._check_action(action, 3, None)

        # action discrete
        self.space.set_action_division(5)
        decode_action = self.tester.check_action_discrete(
            true_n=0,
            action=3,
        )
        self._check_action(decode_action, 3, [3, 3, 3])
        with self.assertRaises(NotImplementedError):
            self.space.action_discrete_encode(decode_action)

        # action_continuous
        decode_action = self.tester.check_action_continuous(
            true_n=3,
            true_low=[-np.inf] * 3,
            true_high=[np.inf] * 3,
            action=[1.1, 0.1, 0.9],
        )
        self._check_action(decode_action, 3, [1.1, 0.1, 0.9])

        # observation discrete
        self.tester.check_observation_discrete(
            true_shape=(3,),
            state=[1.1, 0.1, 0.9],
            encode_state=[1, 0, 1],
        )

        # observation continuous
        self.tester.check_observation_continuous(
            true_shape=(3,),
            state=[1.1, 0.1, 0.9],
            encode_state=np.array([1.1, 0.1, 0.9], dtype=np.float32),
        )

    def test_convert(self):
        space = ArrayContinuousSpace(3, -1, 3)
        val = space.convert(1.2)
        np.testing.assert_array_equal([1.2], val)

        val = space.convert([1.2, 0.9])
        np.testing.assert_array_equal([1.2, 0.9], val)

        val = space.convert((9, 10, True))
        np.testing.assert_array_equal([9.0, 10.0, 1.0], val)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_inf", verbosity=2)
