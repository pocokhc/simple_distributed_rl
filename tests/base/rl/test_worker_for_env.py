import unittest
from typing import Any, cast

import numpy as np
import srl
from srl.base.define import EnvObservationType, Info, RLAction, RLActionType, RLObservationType
from srl.base.env import registration
from srl.base.env.base import EnvBase, EnvRun, SpaceBase
from srl.base.env.genre.singleplay import SinglePlayEnv
from srl.base.env.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.env.spaces.box import BoxSpace
from srl.base.env.spaces.discrete import DiscreteSpace
from srl.base.rl.base import RLConfig, RLWorker
from srl.test.env import TestEnv


class StubEnv(SinglePlayEnv):
    def __init__(self):
        self._action_space: SpaceBase = DiscreteSpace(5)
        self._observation_space: SpaceBase = DiscreteSpace(5)
        self._observation_type = EnvObservationType.UNKNOWN

        self.s_state = np.array(0)
        self.s_reward = 0
        self.s_done = True
        self.s_info = {}
        self.s_actions = [0]

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

    def call_reset(self) -> np.ndarray:
        return self.s_state

    def call_step(self, actions):
        self.s_actions = actions
        return self.s_state, self.s_reward, self.s_done, self.s_info

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
        self.on_reset_state = np.array(0)
        self.state = np.array(0)
        self.action = 0

    def _call_on_reset(
        self,
        state: np.ndarray,
        env: EnvBase,
    ) -> None:
        self.on_reset_state = state

    def _call_policy(
        self,
        state: np.ndarray,
        env: EnvBase,
    ) -> RLAction:
        self.state = state
        return self.action

    def _call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        env: EnvBase,
    ) -> Info:
        self.state = next_state
        return {}

    def _call_render(self, env: EnvBase, player_index: int) -> None:
        raise NotImplementedError()


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.rl_config = StubRLConfig()
        self.env_run = srl.envs.make("Stub")
        self.env = cast(StubEnv, self.env_run.get_original_env())
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

                self.rl_config.set_config_by_env(self.env_run)
                worker = StubRLWorker(self.rl_config)
                self.env_run.reset()
                worker.on_reset(self.env_run, 0)

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
                action = worker.policy(self.env_run)
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
                    env_state = np.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])

                if pat[1] == "Dis":
                    self.rl_config._observation_type = RLObservationType.DISCRETE
                if pat[1] == "Box":
                    self.rl_config._observation_type = RLObservationType.CONTINUOUS
                if pat[1] == "Any":
                    self.rl_config._observation_type = RLObservationType.ANY

                self.rl_config.set_config_by_env(self.env_run)
                worker = StubRLWorker(self.rl_config)
                self.env_run.reset()
                worker.on_reset(self.env_run, 0)

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
                self.env.s_state = env_state
                self.env_run.reset()
                worker.on_reset(self.env_run, 0)
                # policy
                worker.action = rl_action
                action = worker.policy(self.env_run)
                self.assertTrue(isinstance(worker.on_reset_state, np.ndarray))
                self.assertTrue(np.allclose(worker.on_reset_state, rl_state))
                np.testing.assert_array_equal([action], [env_action])
                self.assertTrue(isinstance(worker.state, np.ndarray))
                self.assertTrue(np.allclose(worker.state, rl_state))
                # on_step
                self.env_run.step([action])
                worker.on_step(self.env_run)
                self.assertTrue(isinstance(worker.state, np.ndarray))
                self.assertTrue(np.allclose(worker.state, rl_state))


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_observation", verbosity=2)
