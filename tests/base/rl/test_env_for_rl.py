import itertools
import unittest
from dataclasses import dataclass
from typing import Any, Tuple

import gym
import gym.spaces
import numpy as np
from srl.base.define import EnvObservationType, RLActionType, RLObservationType
from srl.base.env.env import EnvBase, GymEnvWrapper
from srl.base.rl.config import RLConfig
from srl.base.rl.env_for_rl import EnvForRL


@dataclass
class StubEnv(EnvBase):

    action_space_: gym.spaces.Space
    observation_space_: gym.spaces.Space

    def __post_init__(self):
        self.return_state = self.observation_space_.sample()
        self.return_reward = 0
        self.return_done = True
        self.return_info = {}
        self._observation_type = EnvObservationType.UNKOWN

    @property
    def action_space(self) -> gym.spaces.Space:
        return self.action_space_

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self.observation_space_

    @property
    def observation_type(self) -> EnvObservationType:
        return self._observation_type

    @property
    def max_episode_steps(self) -> int:
        return 0

    def reset(self) -> Any:
        return self.return_state

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        return self.return_state, self.return_reward, self.return_done, self.return_info

    def fetch_valid_actions(self) -> Any:
        return None

    def render(self, mode: str = "human") -> Any:
        return

    def backup(self) -> Any:
        pass

    def restore(self, state: Any) -> None:
        pass


@dataclass
class StubConfig(RLConfig):

    action_type_: RLActionType
    observation_type_: RLObservationType

    def __post_init__(self):
        super().__init__(0, 1)

    @staticmethod
    def getName() -> str:
        return "Stub"

    @property
    def action_type(self) -> RLActionType:
        return self.action_type_

    @property
    def observation_type(self) -> RLObservationType:
        return self.observation_type_

    def set_config_by_env(self, env: EnvForRL) -> None:
        pass


class TestEnvForRL_Action(unittest.TestCase):
    def test_at_rl_discrete(self):
        rl_config = StubConfig(RLActionType.DISCRETE, RLObservationType.DISCRETE)
        observation_space = gym.spaces.Box(low=-1, high=3, shape=(1,))

        with self.subTest("Discrete"):
            action_space = gym.spaces.Discrete(5)
            env = EnvForRL(GymEnvWrapper(StubEnv(action_space, observation_space)), rl_config)
            self.assertTrue(env.action_change_type == "")
            self.assertTrue(env.action_space.n == action_space.n)  # type: ignore

        with self.subTest("TupleDiscrete"):
            action_space = gym.spaces.Tuple(
                [
                    gym.spaces.Discrete(2),
                    gym.spaces.Discrete(5),
                    gym.spaces.Discrete(3),
                ]
            )
            env = EnvForRL(GymEnvWrapper(StubEnv(action_space, observation_space)), rl_config)
            self.assertTrue(env.action_change_type == "Tuple->Discrete")
            self.assertTrue(env.action_space.n == 2 * 5 * 3)  # type: ignore

        with self.subTest("Box1"):
            action_division_count = 5
            action_space = gym.spaces.Box(low=-1, high=3, shape=(1,))
            env = EnvForRL(
                GymEnvWrapper(StubEnv(action_space, observation_space)),
                rl_config,
                action_division_count=action_division_count,
            )
            self.assertTrue(env.action_change_type == "Box->Discrete")
            self.assertTrue(env.action_space.n == action_division_count)  # type: ignore
            self.assertTrue(env.action_tbl[0] == [-1])
            self.assertTrue(env.action_tbl[1] == [0])
            self.assertTrue(env.action_tbl[2] == [1])
            self.assertTrue(env.action_tbl[3] == [2])
            self.assertTrue(env.action_tbl[4] == [3])

        with self.subTest("Box2"):
            shape = (3, 2)
            action_space = gym.spaces.Box(low=-1, high=3, shape=shape)
            env = EnvForRL(
                GymEnvWrapper(StubEnv(action_space, observation_space)),
                rl_config,
                action_division_count=action_division_count,
            )
            self.assertTrue(env.action_change_type == "Box->Discrete")
            self.assertTrue(env.action_space.n == 5 ** (3 * 2))  # type: ignore

            _t = list(itertools.product([-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3]))
            true_tbl = list(itertools.product(_t, _t, _t))
            for a in range(len(true_tbl)):
                np.testing.assert_array_equal(env.action_tbl[a], true_tbl[a])

    def test_at_rl_continuous(self):
        rl_config = StubConfig(RLActionType.CONTINUOUS, RLObservationType.DISCRETE)
        observation_space = gym.spaces.Box(low=-1, high=3, shape=(1,))

        with self.subTest("Discrete"):
            action_space = gym.spaces.Discrete(5)
            env = EnvForRL(GymEnvWrapper(StubEnv(action_space, observation_space)), rl_config)
            self.assertTrue(env.action_change_type == "Discrete->Box")
            self.assertTrue(env.action_space.shape == (1,))  # type: ignore
            np.testing.assert_array_equal(env.action_space.low, np.full((1,), 0))  # type: ignore
            np.testing.assert_array_equal(env.action_space.high, np.full((1,), 4))  # type: ignore

        with self.subTest("TupleDiscrete"):
            action_space = gym.spaces.Tuple(
                [
                    gym.spaces.Discrete(2),
                    gym.spaces.Discrete(5),
                    gym.spaces.Discrete(3),
                ]
            )
            env = EnvForRL(GymEnvWrapper(StubEnv(action_space, observation_space)), rl_config)
            self.assertTrue(env.action_change_type == "Tuple->Box")
            self.assertTrue(env.action_space.shape == (3,))  # type: ignore
            np.testing.assert_array_equal(env.action_space.low, [0, 0, 0])  # type: ignore
            np.testing.assert_array_equal(env.action_space.high, [2, 5, 3])  # type: ignore

        with self.subTest("Box"):
            action_space = gym.spaces.Box(low=-1, high=3, shape=(5, 2, 3))
            env = EnvForRL(GymEnvWrapper(StubEnv(action_space, observation_space)), rl_config)
            self.assertTrue(env.action_change_type == "")
            self.assertTrue(env.action_space.shape == action_space.shape)  # type: ignore
            np.testing.assert_array_equal(env.action_space.low, action_space.low)  # type: ignore
            np.testing.assert_array_equal(env.action_space.high, action_space.high)  # type: ignore


class TestEnvForRL_Observation(unittest.TestCase):
    def test_at_rl_discrete(self):
        rl_config = StubConfig(RLActionType.DISCRETE, RLObservationType.DISCRETE)
        action_space = gym.spaces.Discrete(5)

        with self.subTest("Discrete"):
            observation_space = gym.spaces.Discrete(5)
            env = EnvForRL(GymEnvWrapper(StubEnv(action_space, observation_space)), rl_config)

            self.assertTrue(env.observation_change_type == "Discrete->Box")
            self.assertTrue(env.observation_type == EnvObservationType.DISCRETE)
            self.assertTrue(env.observation_space.shape == (1,))  # type: ignore
            np.testing.assert_array_equal(env.observation_space.low, np.full((1,), 0))  # type: ignore
            np.testing.assert_array_equal(env.observation_space.high, np.full((1,), 4))  # type: ignore
            self.assertTrue(env._observation_discrete_diff is None)

        with self.subTest("TupleDiscrete"):
            observation_space = gym.spaces.Tuple(
                [
                    gym.spaces.Discrete(2),
                    gym.spaces.Discrete(5),
                    gym.spaces.Discrete(3),
                ]
            )
            env = EnvForRL(GymEnvWrapper(StubEnv(action_space, observation_space)), rl_config)

            self.assertTrue(env.observation_change_type == "Tuple->Box")
            self.assertTrue(env.observation_type == EnvObservationType.DISCRETE)
            self.assertTrue(env.observation_space.shape == (3,))  # type: ignore
            np.testing.assert_array_equal(env.observation_space.low, [0, 0, 0])  # type: ignore
            np.testing.assert_array_equal(env.observation_space.high, [2, 5, 3])  # type: ignore
            self.assertTrue(env._observation_discrete_diff is None)

        with self.subTest("Box(discrete)"):
            observation_space = gym.spaces.Box(low=-1, high=3, shape=(1,))
            env = StubEnv(action_space, observation_space)
            env.return_state = 1
            prediction_by_simulation = True
            env = EnvForRL(GymEnvWrapper(env), rl_config, prediction_by_simulation=prediction_by_simulation)

            self.assertTrue(env.observation_change_type == "")
            self.assertTrue(env.observation_type == EnvObservationType.DISCRETE)
            self.assertTrue(env.observation_space.shape == observation_space.shape)  # type: ignore
            np.testing.assert_array_equal(env.observation_space.low, observation_space.low)  # type: ignore
            np.testing.assert_array_equal(env.observation_space.high, observation_space.high)  # type: ignore
            self.assertTrue(env._observation_discrete_diff is None)

        with self.subTest("Box(continous)"):
            observation_space = gym.spaces.Box(low=-1, high=4, shape=(2, 2))
            env_org = StubEnv(action_space, observation_space)
            env_org.return_state = [[0, 0.11], [0.99, 1]]
            observation_division_count = 5
            prediction_by_simulation = True
            env = EnvForRL(
                GymEnvWrapper(env_org),
                rl_config,
                prediction_by_simulation=prediction_by_simulation,
                observation_division_count=observation_division_count,
            )

            self.assertTrue(env.observation_change_type == "")
            self.assertTrue(env.observation_type == EnvObservationType.DISCRETE)
            self.assertTrue(env.observation_space.shape == observation_space.shape)  # type: ignore
            np.testing.assert_array_equal(env.observation_space.low, observation_space.low)  # type: ignore
            np.testing.assert_array_equal(env.observation_space.high, observation_space.high)  # type: ignore
            # -1-0 :0
            #  0-1 :1
            #  1-2 :2
            #  2-3 :3
            #  3-4 :4
            np.testing.assert_allclose(env._observation_discrete_diff, [[1.0, 1.0], [1.0, 1.0]])

            # reset
            env_org.return_state = [[0, 0.11], [0.99, 1]]
            true_state = [[1, 1], [1, 2]]
            state = env.reset()
            np.testing.assert_array_equal(true_state, state)

            # step
            env_org.return_state = [[0, 0.11], [0.99, 1]]
            true_state = [[1, 1], [1, 2]]
            state, _, _, _ = env.step(0)
            np.testing.assert_array_equal(true_state, state)


if __name__ == "__main__":
    # unittest.main(module=__name__, defaultTest="TestEnvForRL_Action.test_at_rl_continuous", verbosity=2)
    unittest.main(module=__name__, defaultTest="TestEnvForRL_Observation.test_at_rl_discrete", verbosity=2)
