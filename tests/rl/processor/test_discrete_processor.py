import itertools
import unittest

import gym
import gym.spaces
import numpy as np
from srl.base.define import EnvActionType, EnvObservationType, RLActionType, RLObservationType
from srl.rl.processor.discrete_processor import DiscreteProcessor


class TestAction(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = DiscreteProcessor()

    def test_Discrete(self):
        space = gym.spaces.Discrete(5)
        action_type = EnvActionType.DISCRETE

        # change info
        new_space, new_action_type = self.processor.change_action_info(space, action_type, RLActionType.DISCRETE)
        self.assertTrue(new_space.n == space.n)
        self.assertTrue(new_action_type == EnvActionType.DISCRETE)

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
        action_type = EnvActionType.DISCRETE

        # change info
        new_space, new_action_type = self.processor.change_action_info(space, action_type, RLActionType.DISCRETE)
        self.assertTrue(new_space.n == 2 * 5 * 3)
        true_tbl = list(
            itertools.product(
                [n for n in range(2)],
                [n for n in range(5)],
                [n for n in range(3)],
            )
        )
        np.testing.assert_array_equal(true_tbl, self.processor.action_tbl)
        self.assertTrue(new_action_type == EnvActionType.DISCRETE)

        # decode
        new_action = self.processor.action_decode(3)
        self.assertTrue(new_action == (0, 1, 0))

    def test_Box1(self):
        self.processor.action_division_num = 5
        space = gym.spaces.Box(low=-1, high=3, shape=(1,))
        action_type = EnvActionType.CONTINUOUS

        # change info
        new_space, new_action_type = self.processor.change_action_info(space, action_type, RLActionType.DISCRETE)
        self.assertTrue(new_space.n == 5)
        true_tbl = [
            [-1],
            [0],
            [1],
            [2],
            [3],
        ]
        np.testing.assert_array_equal(true_tbl, self.processor.action_tbl)
        self.assertTrue(new_action_type == EnvActionType.DISCRETE)

        # decode
        new_action = self.processor.action_decode(3)
        self.assertTrue(new_action == [2])

    def test_Box2(self):
        self.processor.action_division_num = 5
        space = gym.spaces.Box(low=-1, high=3, shape=(3, 2))
        action_type = EnvActionType.CONTINUOUS

        # change info
        new_space, new_action_type = self.processor.change_action_info(space, action_type, RLActionType.DISCRETE)
        self.assertTrue(new_space.n == 5 ** (3 * 2))
        _t = list(itertools.product([-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3]))
        true_tbl = list(itertools.product(_t, _t, _t))
        for a in range(len(true_tbl)):
            np.testing.assert_array_equal(true_tbl[a], self.processor.action_tbl[a])
        self.assertTrue(new_action_type == EnvActionType.DISCRETE)

        # decode
        new_action = self.processor.action_decode(3)
        np.testing.assert_array_equal(new_action, true_tbl[3])

    def test_rl_continuous(self):
        # rlがcontinuousは何もしない
        space = gym.spaces.Box(low=-1, high=3, shape=(5, 2, 3))
        action_type = EnvActionType.CONTINUOUS
        new_space, new_action_type = self.processor.change_action_info(space, action_type, RLActionType.CONTINUOUS)
        self.assertTrue(new_space.shape == space.shape)
        np.testing.assert_array_equal(space.low, new_space.low)
        np.testing.assert_array_equal(space.high, new_space.high)
        self.assertTrue(new_action_type == EnvActionType.CONTINUOUS)


class TestObservation(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = DiscreteProcessor()

    def test_Box(self):
        self.processor.observation_division_num = 5

        space = gym.spaces.Box(low=-1, high=4, shape=(2, 2))
        obs_type = EnvObservationType.CONTINUOUS

        # change info
        new_space, new_type = self.processor.change_observation_info(space, obs_type, RLObservationType.DISCRETE)
        self.assertTrue(new_type == EnvObservationType.DISCRETE)
        self.assertTrue(new_space.shape == space.shape)
        np.testing.assert_array_equal(new_space.low, space.low)
        np.testing.assert_array_equal(new_space.high, space.high)
        # -1-0 :0
        #  0-1 :1
        #  1-2 :2
        #  2-3 :3
        #  3-4 :4
        np.testing.assert_allclose(self.processor._observation_discrete_diff, [[1.0, 1.0], [1.0, 1.0]])

        # decode
        new_obs = self.processor.observation_encode(np.asarray([[0, 0.11], [0.99, 1]]))
        np.testing.assert_array_equal(new_obs, [[1, 1], [1, 2]])


if __name__ == "__main__":
    name = "test_Box2"
    unittest.main(module=__name__, defaultTest="Test." + name, verbosity=2)
