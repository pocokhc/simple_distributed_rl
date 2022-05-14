import unittest
from typing import Any, List, Tuple

import numpy as np
from srl.base.define import Action, EnvObservationType, Info, RLActionType, RLObservationType
from srl.base.env import registration
from srl.base.env.base import EnvBase, EnvConfig, SpaceBase
from srl.base.env.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.env.spaces.box import BoxSpace
from srl.base.env.spaces.discrete import DiscreteSpace
from srl.base.rl.base import RLConfig, RLWorker
from srl.test.env import TestEnv


class StubEnv(EnvBase):
    def __init__(self):
        self._action_space: SpaceBase = DiscreteSpace(5)
        self._observation_space: SpaceBase = DiscreteSpace(5)
        self._observation_type = EnvObservationType.UNKNOWN

        self.state = np.array(0)
        self.reward = 0
        self.done = True
        self.info = {}
        self.actions = [0]

    @property
    def action_space(self) -> SpaceBase:
        return self._action_space

    @property
    def observation_space(self) -> SpaceBase:
        return self._observation_space

    @property
    def observation_type(self) -> EnvObservationType:
        return self._observation_type

    @property
    def max_episode_steps(self) -> int:
        return 0

    @property
    def player_num(self) -> int:
        return 1

    def reset(self) -> Tuple[np.ndarray, List[int]]:
        return self.state, [0]

    def step(self, actions: List):
        self.actions = actions
        return self.state, [self.reward], self.done, [0], self.info

    def get_next_player_indices(self) -> List[int]:
        return [0]

    def get_invalid_actions(self, player_index: int):
        return []

    def backup(self) -> Any:
        pass  # do nothing

    def restore(self, state: Any) -> None:
        pass  # do nothing


registration.register(
    id="Stub",
    entry_point=__name__ + ":StubEnv",
)


class StubRLConfig(RLConfig):
    def __init__(self) -> None:
        super().__init__()
        self._action_type = RLActionType.ANY
        self._observation_type = RLObservationType.ANY

    @staticmethod
    def getName() -> str:
        return "Stub"

    @property
    def action_type(self) -> RLActionType:
        return self._action_type

    @property
    def observation_type(self) -> RLObservationType:
        return self._observation_type

    def _set_config_by_env(
        self,
        env: EnvBase,
        env_action_space: SpaceBase,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationType,
    ) -> None:
        pass  # do nothing


class StubRLWorker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.state = np.array(0)
        self.action = 0

    def _on_reset(
        self,
        state: np.ndarray,
        player_index: int,
        env: EnvBase,
    ) -> None:
        self.state = state

    def _policy(
        self,
        state: np.ndarray,
        player_index: int,
        env: EnvBase,
    ) -> Action:
        self.state = state
        return self.action

    def _on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        player_index: int,
        env: EnvBase,
    ) -> Info:
        self.state = next_state
        return {}

    def render(self, env: EnvBase) -> None:
        raise NotImplementedError()


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.rl_config = StubRLConfig()
        self.env = StubEnv()
        self.env_config = EnvConfig("Stub")
        self.rl_config = StubRLConfig()

    def test_env_play(self):
        tester = TestEnv()
        tester.play_test("Stub")

    def test_action(self):
        action_patterns = [
            ["Dis", "Dis", 1, 1],
            ["Dis", "Con", [0.9], 1],
            ["Dis", "Any", 1, 1],
            ["Array", "Dis", 1, [0, 0, 1]],
            ["Array", "Con", [0.1, 0.1, 1.1], [0, 0, 1]],
            ["Array", "Any", [0, 0, 1], [0, 0, 1]],
            ["Box", "Dis", 1, [[-1.0, -1.0, -1.0], [-1.0, -1.0, 0.0]]],
            ["Box", "Con", [0.1, 0.2, 0.0, -0.1, -0.2, -0.01], [[0.1, 0.2, 0.0], [-0.1, -0.2, -0.01]]],
            ["Box", "Any", [[0.1, 0.2, 0.0], [-0.1, -0.2, -0.01]], [[0.1, 0.2, 0.0], [-0.1, -0.2, -0.01]]],
        ]
        self.env._observation_space = DiscreteSpace(5)
        self.env._observation_type = EnvObservationType.DISCRETE
        self.rl_config._observation_type = RLObservationType.DISCRETE
        state = np.array(0)

        for pat in action_patterns:
            with self.subTest(pat):
                rl_action = pat[2]
                env_action = pat[3]
                if pat[0] == "Dis":
                    self.env._action_space = DiscreteSpace(5)
                elif pat[0] == "Array":
                    self.env._action_space = ArrayDiscreteSpace([2, 3, 5])
                elif pat[0] == "Box":
                    self.env._action_space = BoxSpace(low=-1, high=3, shape=(2, 3))

                if pat[1] == "Dis":
                    self.rl_config._action_type = RLActionType.DISCRETE
                elif pat[1] == "Con":
                    self.rl_config._action_type = RLActionType.CONTINUOUS
                elif pat[1] == "Any":
                    self.rl_config._action_type = RLActionType.ANY

                self.rl_config.set_config_by_env(self.env)
                worker = StubRLWorker(self.rl_config)

                action_space = worker.config.env_action_space
                if pat[0] == "Dis":
                    self.assertTrue(isinstance(action_space, DiscreteSpace))
                    self.assertTrue(worker.action_decode(rl_action) == env_action)
                elif pat[0] == "Array":
                    self.assertTrue(isinstance(action_space, ArrayDiscreteSpace))
                    np.testing.assert_array_equal(worker.action_decode(rl_action), env_action)
                elif pat[0] == "Box":
                    self.assertTrue(isinstance(action_space, BoxSpace))
                    np.testing.assert_array_equal(worker.action_decode(rl_action), env_action)

                self.assertTrue(isinstance(worker.config.env_observation_space, DiscreteSpace))
                self.assertTrue(worker.config.env_observation_type == EnvObservationType.DISCRETE)

                worker.action = rl_action
                action = worker.policy(state, 0, self.env)
                np.testing.assert_array_equal([action], [env_action])

    def test_observation(self):
        patterns = [
            ["Dis", "Dis", [5]],
            ["Dis", "Box", [5.0]],
            ["Dis", "Any", 5],
            ["Array", "Dis", [1, 2]],
            ["Array", "Box", [1, 2]],
            ["Array", "Any", [1, 2]],
            ["Box", "Dis", [[0, 0, 0], [1, 1, 1]]],
            ["Box", "Box", [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]]],
            ["Box", "Any", [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]]],
        ]
        self.env._action_space = DiscreteSpace(5)
        self.rl_config._action_type = RLActionType.DISCRETE
        rl_action = 1
        env_action = 1

        for pat in patterns:
            with self.subTest(pat):
                rl_state = np.array(pat[2])

                if pat[0] == "Dis":
                    self.env._observation_space = DiscreteSpace(5)
                    self.env._observation_type = EnvObservationType.DISCRETE
                    env_state = 5
                elif pat[0] == "Array":
                    self.env._observation_space = ArrayDiscreteSpace([2, 3])
                    self.env._observation_type = EnvObservationType.DISCRETE
                    env_state = [1, 2]
                elif pat[0] == "Box":
                    self.env._observation_space = BoxSpace(low=-1, high=3, shape=(2, 3))
                    self.env._observation_type = EnvObservationType.UNKNOWN
                    env_state = [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]]

                if pat[1] == "Dis":
                    self.rl_config._observation_type = RLObservationType.DISCRETE
                if pat[1] == "Box":
                    self.rl_config._observation_type = RLObservationType.CONTINUOUS
                if pat[1] == "Any":
                    self.rl_config._observation_type = RLObservationType.ANY

                self.rl_config.set_config_by_env(self.env)
                worker = StubRLWorker(self.rl_config)

                self.assertTrue(isinstance(worker.config.env_action_space, DiscreteSpace))
                self.assertTrue(worker.action_decode(rl_action) == env_action)

                observation_space = worker.config.env_observation_space
                observation_type = worker.config.env_observation_type
                if pat[0] == "Dis":
                    self.assertTrue(isinstance(observation_space, DiscreteSpace))
                    self.assertTrue(observation_type == EnvObservationType.DISCRETE)
                elif pat[0] == "Array":
                    self.assertTrue(isinstance(observation_space, ArrayDiscreteSpace))
                    self.assertTrue(observation_type == EnvObservationType.DISCRETE)
                elif pat[0] == "Box":
                    self.assertTrue(isinstance(observation_space, BoxSpace))
                    self.assertTrue(observation_type == EnvObservationType.UNKNOWN)

                self.assertTrue(np.allclose(worker.observation_encode(env_state, self.env), rl_state))

                # on_reset
                worker.on_reset(env_state, 0, None)
                self.assertTrue(isinstance(worker.state, np.ndarray))
                self.assertTrue(np.allclose(worker.state, rl_state))
                # policy
                worker.action = rl_action
                action = worker.policy(env_state, 0, None)
                np.testing.assert_array_equal([action], [env_action])
                self.assertTrue(isinstance(worker.state, np.ndarray))
                self.assertTrue(np.allclose(worker.state, rl_state))
                # on_step
                worker.on_step(env_state, 0, True, 0, None)
                self.assertTrue(isinstance(worker.state, np.ndarray))
                self.assertTrue(np.allclose(worker.state, rl_state))


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_observation", verbosity=2)
