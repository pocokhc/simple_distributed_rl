import unittest

import gym
import gym.spaces
import numpy as np
from srl.base.define import EnvObservationType, RLObservationType
from srl.rl.processor.observation_discrete_processor import ObservationDiscreteProcessor


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = ObservationDiscreteProcessor()

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
    name = "test_Box"
    unittest.main(module=__name__, defaultTest="Test." + name, verbosity=2)
