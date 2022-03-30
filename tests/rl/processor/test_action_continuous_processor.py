import itertools
import unittest

import gym
import gym.spaces
import numpy as np
from srl.base.define import RLActionType
from srl.rl.processor.action_continuous_processor import ActionContinuousProcessor


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = ActionContinuousProcessor()

    def test_Discrete(self):
        space = gym.spaces.Discrete(5)

        # change info
        new_space = self.processor.change_action_info(space, RLActionType.CONTINUOUS)
        self.assertTrue(new_space.shape == (1,))
        np.testing.assert_array_equal(new_space.low, np.full((1,), 0))
        np.testing.assert_array_equal(new_space.high, np.full((1,), 4))

        # decode
        new_action = self.processor.action_decode([1.1])
        self.assertTrue(new_action == 1)

    def test_TupleDiscrete(self):
        space = gym.spaces.Tuple(
            [
                gym.spaces.Discrete(2),
                gym.spaces.Discrete(5),
                gym.spaces.Discrete(3),
            ]
        )

        # change info
        new_space = self.processor.change_action_info(space, RLActionType.CONTINUOUS)
        self.assertTrue(new_space.shape == (3,))
        np.testing.assert_array_equal(new_space.low, [0, 0, 0])
        np.testing.assert_array_equal(new_space.high, [1, 4, 2])

        # decode
        new_action = self.processor.action_decode([0.1, 3.9, 1.9])
        self.assertTrue(new_action == [0, 4, 2])

    def test_Box(self):
        space = gym.spaces.Box(low=-1, high=3, shape=(5, 2, 3))

        # change info
        new_space = self.processor.change_action_info(space, RLActionType.CONTINUOUS)
        self.assertTrue(new_space.shape == space.shape)
        np.testing.assert_array_equal(new_space.low, space.low)
        np.testing.assert_array_equal(new_space.high, space.high)

        # decode
        new_action = self.processor.action_decode([0.1, 2.9, 1, 9])
        np.testing.assert_array_equal(new_action, [0.1, 2.9, 1, 9])

    def test_rl_discrete(self):
        # discreteは何もしない
        space = gym.spaces.Discrete(5)
        new_space = self.processor.change_action_info(space, RLActionType.DISCRETE)
        self.assertTrue(new_space.n == space.n)


if __name__ == "__main__":
    name = "test_TupleDiscrete"
    unittest.main(module=__name__, defaultTest="Test." + name, verbosity=2)
