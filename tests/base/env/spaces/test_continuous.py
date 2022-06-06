import itertools
import unittest

import numpy as np
from srl.base.env.spaces import ContinuousSpace
from tests.base.env.space_test import SpaceTest


class Test(unittest.TestCase):
    def _check_action(self, decode_action, true_action):
        self.assertTrue(isinstance(decode_action, float))
        if true_action is not None:
            self.assertTrue(decode_action == true_action)

    def test_space(self):
        self.space = ContinuousSpace(-1, 3)
        self.tester = SpaceTest(self, self.space)

        # sample
        for _ in range(100):
            action = self.space.sample()
            self._check_action(action, None)
            self.assertTrue(action >= -1)
            self.assertTrue(action <= 3)

        # action discrete
        self.space.set_action_division(5)
        true_tbl = [
            [-1],
            [0],
            [1],
            [2],
            [3],
        ]
        np.testing.assert_array_equal(true_tbl[0], self.space.action_tbl[0])
        decode_action = self.tester.check_action_discrete(
            true_n=5,
            action=3,
        )
        self._check_action(decode_action, 2)

        # action_continuous
        decode_action = self.tester.check_action_continuous(
            true_n=1,
            true_low=[-1],
            true_high=[3],
            action=[1.1],
        )
        self._check_action(decode_action, 1.1)

        # observation discrete
        self.tester.check_observation_discrete(
            true_shape=(1,),
            true_low=[-1],
            true_high=[3],
            state=1.1,
            encode_state=[1],
        )

        # observation continuous
        self.tester.check_observation_continuous(
            true_shape=(1,),
            true_low=[-1],
            true_high=[3],
            state=1.1,
            encode_state=[1.1],
        )

        # eq
        self.assertTrue(self.space == ContinuousSpace(-1, 3))
        self.assertTrue(self.space != ContinuousSpace(-1, 2))

    def test_inf(self):
        self.space = ContinuousSpace()
        self.tester = SpaceTest(self, self.space)

        # sample
        for _ in range(100):
            action = self.space.sample()
            self._check_action(action, None)

        # action discrete
        self.space.set_action_division(5)
        decode_action = self.tester.check_action_discrete(
            true_n=0,
            action=3,
        )
        self._check_action(decode_action, 3)

        # action_continuous
        decode_action = self.tester.check_action_continuous(
            true_n=1,
            true_low=[-np.inf],
            true_high=[np.inf],
            action=[1.1],
        )
        self._check_action(decode_action, 1.1)

        # observation discrete
        self.tester.check_observation_discrete(
            true_shape=(1,),
            true_low=[-np.inf],
            true_high=[np.inf],
            state=1.1,
            encode_state=[1],
        )

        # observation continuous
        self.tester.check_observation_continuous(
            true_shape=(1,),
            true_low=[-np.inf],
            true_high=[np.inf],
            state=1.1,
            encode_state=[1.1],
        )


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_inf", verbosity=2)
