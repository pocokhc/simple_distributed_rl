import unittest

import gym
import gym.spaces
import numpy as np
from srl.base.define import EnvActionType, EnvObservationType, RLActionType
from srl.base.env.processors import ContinuousProcessor


class TestAction(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = ContinuousProcessor()

    def test_Discrete(self):
        space = gym.spaces.Discrete(5)
        action_type = EnvActionType.DISCRETE

        # change info
        new_space, new_action_type = self.processor.change_action_info(
            space, action_type, RLActionType.CONTINUOUS, None
        )
        self.assertTrue(new_space.shape == (1,))
        np.testing.assert_array_equal(new_space.low, np.full((1,), 0))
        np.testing.assert_array_equal(new_space.high, np.full((1,), 4))
        self.assertTrue(new_action_type == EnvActionType.CONTINUOUS)

        # decode
        new_action = self.processor.action_decode([1.1], None)
        self.assertTrue(new_action == 1)

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
        new_space, new_action_type = self.processor.change_action_info(
            space, action_type, RLActionType.CONTINUOUS, None
        )
        self.assertTrue(new_space.shape == (3,))
        np.testing.assert_array_equal(new_space.low, [0, 0, 0])
        np.testing.assert_array_equal(new_space.high, [1, 4, 2])
        self.assertTrue(new_action_type == EnvActionType.CONTINUOUS)

        # decode
        new_action = self.processor.action_decode([0.1, 3.9, 1.9], None)
        self.assertTrue(new_action == [0, 4, 2])

    def test_Box(self):
        space = gym.spaces.Box(low=-1, high=3, shape=(5, 2, 3))
        action_type = EnvActionType.DISCRETE

        # change info
        new_space, new_action_type = self.processor.change_action_info(
            space, action_type, RLActionType.CONTINUOUS, None
        )
        self.assertTrue(new_space.shape == space.shape)
        np.testing.assert_array_equal(new_space.low, space.low)
        np.testing.assert_array_equal(new_space.high, space.high)
        self.assertTrue(new_action_type == EnvActionType.CONTINUOUS)

        # decode
        new_action = self.processor.action_decode([0.1, 2.9, 1, 9], None)
        np.testing.assert_array_equal(new_action, [0.1, 2.9, 1, 9])

    def test_rl_discrete(self):
        # rl が discrete は何もしない
        space = gym.spaces.Discrete(5)
        action_type = EnvActionType.DISCRETE
        new_space, new_action_type = self.processor.change_action_info(space, action_type, RLActionType.DISCRETE, None)
        self.assertTrue(new_space.n == space.n)
        self.assertTrue(new_action_type == EnvActionType.DISCRETE)

        # decode
        new_action = self.processor.action_decode([0, 4, 2], None)
        np.testing.assert_array_equal(new_action, [0, 4, 2])

    def test_rl_any(self):
        # rl が any は何もしない
        space = gym.spaces.Discrete(5)
        action_type = EnvActionType.DISCRETE
        new_space, new_action_type = self.processor.change_action_info(space, action_type, RLActionType.ANY, None)
        self.assertTrue(new_space.n == space.n)
        self.assertTrue(new_action_type == EnvActionType.DISCRETE)

        # decode
        new_action = self.processor.action_decode([0, 4, 2], None)
        np.testing.assert_array_equal(new_action, [0, 4, 2])


class TestObservation(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = ContinuousProcessor()

    def test_rl_any(self):
        # rl が any は何もしない
        space = gym.spaces.Box(low=-1, high=4, shape=(2, 2))
        obs_type = EnvObservationType.DISCRETE

        new_space, new_type = self.processor.change_observation_info(space, obs_type, RLActionType.ANY, None)
        self.assertTrue(new_type == EnvObservationType.DISCRETE)
        self.assertTrue(new_space.shape == space.shape)
        np.testing.assert_array_equal(new_space.low, space.low)
        np.testing.assert_array_equal(new_space.high, space.high)

        # decode
        new_obs = self.processor.observation_encode(np.asarray([[0, 1], [2, 3]]), None)
        np.testing.assert_array_equal(new_obs, [[0, 1], [2, 3]])


if __name__ == "__main__":
    name = "test_TupleDiscrete"
    unittest.main(module=__name__, defaultTest="Test." + name, verbosity=2)
