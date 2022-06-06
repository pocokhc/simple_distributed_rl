import unittest

import numpy as np
from srl.base.env.spaces import DiscreteSpace
from tests.base.env.space_test import SpaceTest


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.space = DiscreteSpace(5)
        self.assertTrue(self.space.n == 5)

        self.tester = SpaceTest(self, self.space)

    def _check_action(self, decode_action, true_action):
        self.assertTrue(isinstance(decode_action, int))
        self.assertTrue(decode_action == true_action)

    def test_space(self):
        # sample
        actions = [self.space.sample([3]) for _ in range(100)]
        actions = sorted(list(set(actions)))
        np.testing.assert_array_equal(actions, [0, 1, 2, 4])

        # action discrete
        decode_action = self.tester.check_action_discrete(5, action=2)
        self._check_action(decode_action, 2)
        self.tester.check_action_encode(3, 3)

        # action_continuous
        decode_action = self.tester.check_action_continuous(
            true_n=1,
            true_low=[0],
            true_high=[4],
            action=[3.3],
        )
        self._check_action(decode_action, 3)

        # observation discrete
        self.tester.check_observation_discrete(
            true_shape=(1,),
            true_low=[0],
            true_high=[4],
            state=2,
            encode_state=[2],
        )

        # observation continuous
        self.tester.check_observation_continuous(
            true_shape=(1,),
            true_low=[0],
            true_high=[4],
            state=2,
            encode_state=[2.0],
        )

        # eq
        self.assertTrue(self.space == DiscreteSpace(5))
        self.assertTrue(self.space != DiscreteSpace(4))


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_space", verbosity=2)
