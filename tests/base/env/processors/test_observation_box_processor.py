import unittest
from typing import Any, List, Tuple

import gym
import gym.spaces
import numpy as np
import srl
from srl.base.define import EnvActionType, EnvObservationType, RLObservationType
from srl.base.env.genre.singleplay import SingleActionDiscrete
from srl.base.env.processors import ObservationBoxProcessor
from srl.base.env.registration import register


class StubEnv(SingleActionDiscrete):
    def __init__(self, return_state):
        self.return_state: Any = return_state
        self.return_reward: float = 0
        self.return_done: bool = True
        self.return_info: Any = {}

    @property
    def action_num(self) -> int:
        return len(5)

    @property
    def observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(5)

    @property
    def observation_type(self) -> EnvObservationType:
        return EnvObservationType.UNKOWN

    @property
    def max_episode_steps(self) -> int:
        return 10

    def reset_single(self) -> np.ndarray:
        return self.return_state

    def step_single(self, action_: int) -> Tuple[Any, float, bool, dict]:
        return self.return_state, self.return_reward, self.return_done, self.return_info

    def render_terminal(self):
        pass

    def backup(self):
        pass

    def restore(self, state) -> None:
        pass


register("StubEnv_continuous", __name__ + ":StubEnv", {"return_state": [[1.1, 2.2], [2.2, 3.3]]})
register("StubEnv_discrete", __name__ + ":StubEnv", {"return_state": [[1, 2], [2, 3]]})


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = ObservationBoxProcessor()
        self.obs_type = EnvObservationType.UNKOWN
        self.rl_obs_type = RLObservationType.UNKOWN

    def test_Discrete(self):
        space = gym.spaces.Discrete(5)

        # change info
        new_space, new_type = self.processor.change_observation_info(space, self.obs_type, self.rl_obs_type)
        self.assertTrue(new_type == EnvObservationType.DISCRETE)
        self.assertTrue(new_space.shape == (1,))
        np.testing.assert_array_equal(new_space.low, np.full((1,), 0))
        np.testing.assert_array_equal(new_space.high, np.full((1,), 4))

        # encode
        new_obs = self.processor.observation_encode(1)
        self.assertTrue(isinstance(new_obs, np.ndarray))
        self.assertTrue(new_obs == [1])
        self.assertTrue(new_obs.shape == (1,))

    def test_TupleDiscrete(self):
        space = gym.spaces.Tuple(
            [
                gym.spaces.Discrete(2),
                gym.spaces.Discrete(5),
                gym.spaces.Discrete(3),
            ]
        )

        # change info
        new_space, new_type = self.processor.change_observation_info(space, self.obs_type, self.rl_obs_type)
        self.assertTrue(new_type == EnvObservationType.DISCRETE)
        self.assertTrue(new_space.shape == (3,))
        np.testing.assert_array_equal(new_space.low, [0, 0, 0])
        np.testing.assert_array_equal(new_space.high, [1, 4, 2])

        # decode
        new_obs = self.processor.observation_encode([1, 2, 2])
        self.assertTrue(isinstance(new_obs, np.ndarray))
        np.testing.assert_array_equal(new_obs, [1, 2, 2])
        self.assertTrue(new_obs.shape == (3,))

    def test_Box(self):
        self.processor.prediction_by_simulation = False
        space = gym.spaces.Box(low=-1, high=4, shape=(2, 2))

        # change info
        new_space, new_type = self.processor.change_observation_info(space, self.obs_type, self.rl_obs_type)
        self.assertTrue(new_type == EnvObservationType.UNKOWN)
        self.assertTrue(new_space.shape == space.shape)
        np.testing.assert_array_equal(new_space.low, space.low)
        np.testing.assert_array_equal(new_space.high, space.high)

        # decode
        new_obs = self.processor.observation_encode([[1.1, 2.2], [2.2, 3.3]])
        self.assertTrue(isinstance(new_obs, np.ndarray))
        np.testing.assert_array_equal(new_obs, [[1.1, 2.2], [2.2, 3.3]])
        self.assertTrue(new_obs.shape == (2, 2))

    def test_Box_continuous(self):
        self.processor.prediction_by_simulation = True
        self.processor.env_config = srl.envs.Config("StubEnv_continuous")

        space = gym.spaces.Box(low=-1, high=4, shape=(2, 2))

        # change info
        new_space, new_type = self.processor.change_observation_info(space, self.obs_type, self.rl_obs_type)
        self.assertTrue(new_type == EnvObservationType.CONTINUOUS)
        self.assertTrue(new_space.shape == space.shape)
        np.testing.assert_array_equal(new_space.low, space.low)
        np.testing.assert_array_equal(new_space.high, space.high)

    def test_Box_discrete(self):
        self.processor.prediction_by_simulation = True
        self.processor.env_config = srl.envs.Config("StubEnv_discrete")

        space = gym.spaces.Box(low=-1, high=4, shape=(2, 2))

        # change info
        new_space, new_type = self.processor.change_observation_info(space, self.obs_type, self.rl_obs_type)
        self.assertTrue(new_type == EnvObservationType.DISCRETE)
        self.assertTrue(new_space.shape == space.shape)
        np.testing.assert_array_equal(new_space.low, space.low)
        np.testing.assert_array_equal(new_space.high, space.high)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_Box_discrete", verbosity=2)
