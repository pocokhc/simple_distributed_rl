from typing import Any, List, Tuple, cast

import numpy as np
import pytest

import srl
from srl.base.define import EnvObservationTypes, InfoType, RLActionType, RLActionTypes, RLObservationTypes
from srl.base.env.base import EnvBase, SpaceBase
from srl.base.env.genre.singleplay import SinglePlayEnv
from srl.base.env.registration import register as register_env
from srl.base.rl.config import RLConfig
from srl.base.rl.registration import register as register_rl
from srl.base.rl.worker import RLWorker
from srl.base.spaces import ArrayDiscreteSpace, BoxSpace, DiscreteSpace
from srl.test.env import TestEnv


class StubEnv(SinglePlayEnv):
    def __init__(self):
        self._action_space: SpaceBase = DiscreteSpace(5)
        self._observation_space: SpaceBase = DiscreteSpace(5)
        self._observation_type = EnvObservationTypes.UNKNOWN

        self.s_state: Any = 0
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
    def observation_type(self) -> EnvObservationTypes:
        return self._observation_type

    @property
    def max_episode_steps(self) -> int:
        return 0

    @property
    def player_num(self) -> int:
        return 1

    def call_reset(self) -> Tuple[int, InfoType]:
        return self.s_state, {}

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
        self._action_type = RLActionTypes.ANY
        self._observation_type = RLObservationTypes.ANY

    def getName(self) -> str:
        return "Stub"

    @property
    def action_type(self) -> RLActionTypes:
        return self._action_type

    @property
    def observation_type(self) -> RLObservationTypes:
        return self._observation_type

    def set_config_by_env(
        self,
        env: EnvBase,
        env_action_space: SpaceBase,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationTypes,
    ) -> None:
        pass  # do nothing


class StubRLWorker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.on_reset_state = np.array(0)
        self.state = np.array(0)
        self.action = 0

    def _call_on_reset(self, state: np.ndarray, env: EnvBase, worker) -> dict:
        self.on_reset_state = state
        return {}

    def _call_policy(self, state: np.ndarray, env: EnvBase, worker) -> Tuple[RLActionType, dict]:
        self.state = state
        return self.action, {}

    def _call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        env: EnvBase,
        worker,
    ) -> InfoType:
        self.state = next_state
        return {}


register_rl(StubRLConfig(), "", "", "", __name__ + ":StubRLWorker")


def test_env_play():
    tester = TestEnv()
    tester.play_test("Stub")


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


@pytest.mark.parametrize("env_action_space, rl_action_type, rl_action, env_action", action_patterns)
def test_action(env_action_space, rl_action_type, rl_action, env_action):
    env = srl.make_env("Stub")
    env_org = cast(StubEnv, env.get_original_env())

    rl_config = StubRLConfig()
    rl_config.reset(env)
    worker_run = srl.make_worker(rl_config, env=env)

    env_org._observation_space = DiscreteSpace(5)
    env_org._observation_type = EnvObservationTypes.DISCRETE
    rl_config._observation_type = RLObservationTypes.DISCRETE

    if env_action_space == "Dis":
        env_org._action_space = DiscreteSpace(5)
    elif env_action_space == "Array":
        env_org._action_space = ArrayDiscreteSpace(3, 0, [2, 3, 5])
    elif env_action_space == "Box":
        env_org._action_space = BoxSpace(low=-1, high=3, shape=(2, 3))

    if rl_action_type == "Dis":
        rl_config._action_type = RLActionTypes.DISCRETE
    elif rl_action_type == "Con":
        rl_config._action_type = RLActionTypes.CONTINUOUS
    elif rl_action_type == "Any":
        rl_config._action_type = RLActionTypes.ANY

    env.reset()
    rl_config._is_set_env_config = False
    rl_config.reset(env)
    worker_run.on_reset(env, 0)
    worker = cast(StubRLWorker, worker_run.worker)

    action_space = rl_config.action_space
    if env_action_space == "Dis":
        assert isinstance(action_space, DiscreteSpace)
        assert worker.action_decode(rl_action) == env_action
    elif env_action_space == "Array":
        assert isinstance(action_space, ArrayDiscreteSpace)
        a = cast(List[int], worker.action_decode(rl_action))
        np.testing.assert_array_equal(a, env_action)
    elif env_action_space == "Box":
        assert isinstance(action_space, BoxSpace)
        a = cast(np.ndarray, worker.action_decode(rl_action))
        np.testing.assert_array_equal(a, env_action)

    assert isinstance(rl_config.observation_space, DiscreteSpace)
    assert rl_config.env_observation_type == EnvObservationTypes.DISCRETE

    worker.action = rl_action
    action = cast(int, worker_run.policy(env))
    np.testing.assert_array_equal([action], [env_action])


observation_patterns = [
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


@pytest.mark.parametrize("env_action_space, rl_action_type, rl_state", observation_patterns)
def test_observation(env_action_space, rl_action_type, rl_state):
    env = srl.make_env("Stub")
    env_org = cast(StubEnv, env.get_original_env())
    env_org._action_space = DiscreteSpace(5)

    rl_config = StubRLConfig()
    rl_config.reset(env)
    worker_run = srl.make_worker(rl_config, env=env)

    rl_config._action_type = RLActionTypes.DISCRETE
    rl_action = 1
    env_action = 1

    rl_state = np.array(rl_state)

    env_state = 0
    if env_action_space == "Dis":
        env_org._observation_space = DiscreteSpace(5)
        env_org._observation_type = EnvObservationTypes.DISCRETE
        env_state = 5
    elif env_action_space == "Array":
        env_org._observation_space = ArrayDiscreteSpace(2, 0, [2, 3])
        env_org._observation_type = EnvObservationTypes.DISCRETE
        env_state = [1, 2]
    elif env_action_space == "Box":
        env_org._observation_space = BoxSpace(low=-1, high=3, shape=(2, 3))
        env_org._observation_type = EnvObservationTypes.UNKNOWN
        env_state = np.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])

    if rl_action_type == "Dis":
        rl_config._observation_type = RLObservationTypes.DISCRETE
    if rl_action_type == "Box":
        rl_config._observation_type = RLObservationTypes.CONTINUOUS
    if rl_action_type == "Any":
        rl_config._observation_type = RLObservationTypes.ANY

    env.reset()
    rl_config._is_set_env_config = False
    rl_config.reset(env)
    worker_run.on_reset(env, 0)
    worker = cast(StubRLWorker, worker_run.worker)

    assert isinstance(rl_config.action_space, DiscreteSpace)
    assert worker.action_decode(rl_action) == env_action

    observation_space = rl_config.observation_space
    observation_type = rl_config.env_observation_type
    if env_action_space == "Dis":
        assert isinstance(observation_space, DiscreteSpace)
        assert observation_type == EnvObservationTypes.DISCRETE
    elif env_action_space == "Array":
        assert isinstance(observation_space, ArrayDiscreteSpace)
        assert observation_type == EnvObservationTypes.DISCRETE
    elif env_action_space == "Box":
        assert isinstance(observation_space, BoxSpace)
        assert observation_type == EnvObservationTypes.UNKNOWN

    assert np.allclose(worker.state_encode(env_state, env), rl_state)

    # on_reset
    env_org.s_state = env_state
    env.reset()
    worker_run.on_reset(env, 0)
    # policy
    worker.action = rl_action
    action = cast(int, worker_run.policy(env))
    assert isinstance(worker.on_reset_state, np.ndarray)
    assert np.allclose(worker.on_reset_state, rl_state)
    np.testing.assert_array_equal([action], [env_action])
    assert isinstance(worker.state, np.ndarray)
    assert np.allclose(worker.state, rl_state)
    # on_step
    env.step([action])
    worker_run.on_step(env)
    assert isinstance(worker.state, np.ndarray)
    assert np.allclose(worker.state, rl_state)
