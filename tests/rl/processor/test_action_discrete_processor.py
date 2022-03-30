import itertools
import unittest

import gym
import gym.spaces
import numpy as np
from srl.base.define import RLActionType
from srl.rl.processor.action_discrete_processor import ActionDiscreteProcessor


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = ActionDiscreteProcessor()

    def test_Discrete(self):
        space = gym.spaces.Discrete(5)

        # change info
        new_space = self.processor.change_action_info(space, RLActionType.DISCRETE)
        self.assertTrue(new_space.n == space.n)

        # decode
        new_action = self.processor.action_decode(3)
        self.assertTrue(new_action == 3)

    def test_TupleDiscrete(self):
        space = gym.spaces.Tuple(
            [
                gym.spaces.Discrete(2),
                gym.spaces.Discrete(5),
                gym.spaces.Discrete(3),
            ]
        )

        # change info
        new_space = self.processor.change_action_info(space, RLActionType.DISCRETE)
        self.assertTrue(new_space.n == 2 * 5 * 3)
        true_tbl = list(
            itertools.product(
                [n for n in range(2)],
                [n for n in range(5)],
                [n for n in range(3)],
            )
        )
        np.testing.assert_array_equal(true_tbl, self.processor.action_tbl)

        # decode
        new_action = self.processor.action_decode(3)
        self.assertTrue(new_action == (0, 1, 0))

    def test_Box1(self):
        self.processor.action_division_num = 5
        space = gym.spaces.Box(low=-1, high=3, shape=(1,))

        # change info
        new_space = self.processor.change_action_info(space, RLActionType.DISCRETE)
        self.assertTrue(new_space.n == 5)
        true_tbl = [
            [-1],
            [0],
            [1],
            [2],
            [3],
        ]
        np.testing.assert_array_equal(true_tbl, self.processor.action_tbl)

        # decode
        new_action = self.processor.action_decode(3)
        self.assertTrue(new_action == [2])

    def test_Box2(self):
        self.processor.action_division_num = 5
        space = gym.spaces.Box(low=-1, high=3, shape=(3, 2))

        # change info
        new_space = self.processor.change_action_info(space, RLActionType.DISCRETE)
        self.assertTrue(new_space.n == 5 ** (3 * 2))
        _t = list(itertools.product([-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3]))
        true_tbl = list(itertools.product(_t, _t, _t))
        for a in range(len(true_tbl)):
            np.testing.assert_array_equal(true_tbl[a], self.processor.action_tbl[a])

        # decode
        new_action = self.processor.action_decode(3)
        np.testing.assert_array_equal(new_action, true_tbl[3])

    def test_rl_continuous(self):
        # continuousは何もしない
        space = gym.spaces.Box(low=-1, high=3, shape=(5, 2, 3))
        new_space = self.processor.change_action_info(space, RLActionType.CONTINUOUS)
        self.assertTrue(new_space.shape == space.shape)
        np.testing.assert_array_equal(space.low, new_space.low)
        np.testing.assert_array_equal(space.high, new_space.high)


if __name__ == "__main__":
    name = "test_Box2"
    unittest.main(module=__name__, defaultTest="Test." + name, verbosity=2)
