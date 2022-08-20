import unittest
from typing import Any, cast

import numpy as np
import srl
from srl.base.define import EnvObservationType, Info, RLAction, RLActionType, RLObservationType
from srl.base.env.base import EnvBase, SpaceBase
from srl.base.env.genre.singleplay import SinglePlayEnv
from srl.base.env.registration import register as register_env
from srl.base.env.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.env.spaces.box import BoxSpace
from srl.base.env.spaces.discrete import DiscreteSpace
from srl.base.rl.base import RLConfig, RLWorker, WorkerRun
from srl.base.rl.registration import register as register_rl
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
        self.s_action = 0

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

    def call_step(self, action):
        self.s_action = action
        return self.s_state, self.s_reward, self.s_done, self.s_info

    def backup(self, **kwargs) -> Any:
        return None

    def restore(self, state: Any, **kwargs) -> None:
        pass  # do nothing


register_env(
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

    def set_config_by_env(
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

    def _call_on_reset(self, state: np.ndarray, env: EnvBase, worker: WorkerRun) -> None:
        self.on_reset_state = state

    def _call_policy(self, state: np.ndarray, env: EnvBase, worker: WorkerRun) -> RLAction:
        self.state = state
        return self.action

    def _call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        env: EnvBase,
        worker: WorkerRun,
    ) -> Info:
        self.state = next_state
        return {}


register_rl(StubRLConfig, "", "", "", __name__ + ":StubRLWorker")


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.env_run = srl.envs.make("Stub")
        self.env = cast(StubEnv, self.env_run.get_original_env())

        self.rl_config = StubRLConfig()
        self.worker_run = srl.rl.make_worker(self.rl_config, self.env_run)

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

                self.env_run.reset()
                self.rl_config._is_set_env_config = False
                self.rl_config.reset_config(self.env_run)
                self.worker_run.on_reset(self.env_run, 0)
                worker = cast(RLWorker, self.worker_run.worker)

                action_space = self.rl_config.action_space
                if pat[0] == "Dis":
                    self.assertTrue(isinstance(action_space, DiscreteSpace))
                    self.assertTrue(worker.action_decode(rl_action) == env_action)
                elif pat[0] == "Array":
                    self.assertTrue(isinstance(action_space, ArrayDiscreteSpace))
                    np.testing.assert_array_equal(worker.action_decode(rl_action), env_action)
                elif pat[0] == "Box":
                    self.assertTrue(isinstance(action_space, BoxSpace))
                    np.testing.assert_array_equal(worker.action_decode(rl_action), env_action)

                self.assertTrue(isinstance(self.rl_config.observation_space, DiscreteSpace))
                self.assertTrue(self.rl_config.env_observation_type == EnvObservationType.DISCRETE)

                worker.action = rl_action
                action = self.worker_run.policy(self.env_run)
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

                self.env_run.reset()
                self.rl_config._is_set_env_config = False
                self.rl_config.reset_config(self.env_run)
                self.worker_run.on_reset(self.env_run, 0)
                worker = cast(RLWorker, self.worker_run.worker)

                self.assertTrue(isinstance(self.rl_config.action_space, DiscreteSpace))
                self.assertTrue(worker.action_decode(rl_action) == env_action)

                observation_space = self.rl_config.observation_space
                observation_type = self.rl_config.env_observation_type
                if pat[0] == "Dis":
                    self.assertTrue(isinstance(observation_space, DiscreteSpace))
                    self.assertTrue(observation_type == EnvObservationType.DISCRETE)
                elif pat[0] == "Array":
                    self.assertTrue(isinstance(observation_space, ArrayDiscreteSpace))
                    self.assertTrue(observation_type == EnvObservationType.DISCRETE)
                elif pat[0] == "Box":
                    self.assertTrue(isinstance(observation_space, BoxSpace))
                    self.assertTrue(observation_type == EnvObservationType.UNKNOWN)

                self.assertTrue(np.allclose(worker.state_encode(env_state, self.env), rl_state))

                # on_reset
                self.env.s_state = env_state
                self.env_run.reset()
                self.worker_run.on_reset(self.env_run, 0)
                # policy
                worker.action = rl_action
                action = self.worker_run.policy(self.env_run)
                self.assertTrue(isinstance(worker.on_reset_state, np.ndarray))
                self.assertTrue(np.allclose(worker.on_reset_state, rl_state))
                np.testing.assert_array_equal([action], [env_action])
                self.assertTrue(isinstance(worker.state, np.ndarray))
                self.assertTrue(np.allclose(worker.state, rl_state))
                # on_step
                self.env_run.step([action])
                self.worker_run.on_step(self.env_run)
                self.assertTrue(isinstance(worker.state, np.ndarray))
                self.assertTrue(np.allclose(worker.state, rl_state))


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_action", verbosity=2)
