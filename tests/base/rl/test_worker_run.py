from typing import Any, Tuple, cast

import numpy as np
import pytest

import srl
from srl.base.define import EnvObservationTypes, InfoType, RLActionType, RLBaseTypes, RLTypes
from srl.base.env.base import SpaceBase
from srl.base.env.genre.singleplay import SinglePlayEnv
from srl.base.env.registration import register as register_env
from srl.base.rl.base import RLWorker
from srl.base.rl.config import RLConfig
from srl.base.rl.registration import register as register_rl
from srl.base.rl.worker_run import WorkerRun
from srl.base.spaces import ArrayDiscreteSpace, BoxSpace, DiscreteSpace
from srl.base.spaces.array_continuous import ArrayContinuousSpace
from srl.base.spaces.continuous import ContinuousSpace
from srl.test.env import TestEnv


class StubEnv(SinglePlayEnv):
    def __init__(self):
        self._action_space: SpaceBase = DiscreteSpace(5)
        self._observation_space: SpaceBase = DiscreteSpace(5)
        self._observation_type = EnvObservationTypes.UNKNOWN

        self.s_state: Any = 0
        self.s_reward = 0.0
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
        self._action_type = RLBaseTypes.ANY
        self._observation_type = RLBaseTypes.ANY

    def getName(self) -> str:
        return "Stub"

    @property
    def base_action_type(self) -> RLBaseTypes:
        return self._action_type

    @property
    def base_observation_type(self) -> RLBaseTypes:
        return self._observation_type

    def get_use_framework(self) -> str:
        return ""


class StubRLWorker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.on_reset_state = np.array(0)
        self.state = np.array(0)
        self.action = 0

    def on_reset(self, worker: WorkerRun) -> dict:
        self.on_reset_state = worker.state
        return {}

    def policy(self, worker: WorkerRun) -> Tuple[RLActionType, dict]:
        self.state = worker.state
        return self.action, {}

    def on_step(self, worker: WorkerRun) -> InfoType:
        self.state = worker.state
        return {}


register_rl(StubRLConfig(), "", "", "", __name__ + ":StubRLWorker")


def test_env_play():
    tester = TestEnv()
    tester.play_test("Stub")


@pytest.mark.parametrize(
    "rl_action_type, rl_action, env_action_space, true_env_action",
    [
        [RLBaseTypes.DISCRETE, 1, DiscreteSpace(5), 1],
        [RLBaseTypes.DISCRETE, 1, ArrayDiscreteSpace(2, 0, 5), [0, 1]],
        [RLBaseTypes.DISCRETE, 1, ContinuousSpace(0, 5), 1.25],
        [RLBaseTypes.DISCRETE, 1, ArrayContinuousSpace(1, 0, 5), [1.25]],
        [RLBaseTypes.DISCRETE, 2, BoxSpace((1,)), None],
        [RLBaseTypes.DISCRETE, 1, BoxSpace((1,), -1, 1), [-0.5]],
        [RLBaseTypes.CONTINUOUS, 1.2, DiscreteSpace(5), 1],
        [RLBaseTypes.CONTINUOUS, 1.2, ArrayDiscreteSpace(2, 0, 5), [1]],
        [RLBaseTypes.CONTINUOUS, 1.2, ContinuousSpace(0, 5), 1.2],
        [RLBaseTypes.CONTINUOUS, 1.2, ArrayContinuousSpace(1, 0, 5), [1.2]],
        [RLBaseTypes.CONTINUOUS, 1.2, BoxSpace((1,)), [1.2]],
        [RLBaseTypes.CONTINUOUS, [1.2, 2.2], DiscreteSpace(5), 1],
        [RLBaseTypes.CONTINUOUS, [1.2, 2.2], ArrayDiscreteSpace(2, 0, 5), [1, 2]],
        [RLBaseTypes.CONTINUOUS, [1.2, 2.2], ContinuousSpace(0, 5), 1.2],
        [RLBaseTypes.CONTINUOUS, [1.2, 2.2], ArrayContinuousSpace(1, 0, 5), [1.2, 2.2]],
        [RLBaseTypes.CONTINUOUS, [1.2, 2.2], BoxSpace((2, 1)), [[1.2], [2.2]]],
        [RLBaseTypes.ANY, 1, DiscreteSpace(5), 1],
        [RLBaseTypes.ANY, 1, ArrayDiscreteSpace(2, 0, 5), [0, 1]],
        [RLBaseTypes.ANY, 1.2, ContinuousSpace(0, 5), 1.2],
        [RLBaseTypes.ANY, [1.2, 2.2], ArrayContinuousSpace(1, 0, 5), [1.2, 2.2]],
        [RLBaseTypes.ANY, [1.2, 2.2], BoxSpace((2, 1)), [[1.2], [2.2]]],
    ],
)
def test_action_decode(rl_action_type, rl_action, env_action_space, true_env_action):
    print(rl_action_type, rl_action, env_action_space, true_env_action)
    env = srl.make_env("Stub")
    env_org = cast(StubEnv, env.unwrapped)
    env_org._action_space = env_action_space

    rl_config = StubRLConfig()
    rl_config._action_type = rl_action_type

    # ---
    if true_env_action is None:
        with pytest.raises(AssertionError):
            worker_run = srl.make_worker(rl_config, env)
        return
    else:
        worker_run = srl.make_worker(rl_config, env)

    env.reset()
    worker_run.on_reset(0, training=False)
    worker = cast(StubRLWorker, worker_run.worker)

    worker.action = rl_action
    env_action = worker_run.policy()
    print(env_action)
    if isinstance(env_action, list):
        np.testing.assert_array_equal(env_action, true_env_action)
    elif isinstance(env_action, np.ndarray):
        assert (env_action == true_env_action).all()
    else:
        assert env_action == true_env_action


@pytest.mark.parametrize(
    "env_obs_space, env_state, rl_obs_type, true_obs_type, true_state",
    [
        [DiscreteSpace(5), 1, RLBaseTypes.DISCRETE, RLTypes.DISCRETE, 1],
        [DiscreteSpace(5), 1, RLBaseTypes.CONTINUOUS, RLTypes.CONTINUOUS, 1],
        [DiscreteSpace(5), 1, RLBaseTypes.ANY, RLTypes.DISCRETE, 1],
        [ArrayDiscreteSpace(2, 0, 5), [0, 1], RLBaseTypes.DISCRETE, RLTypes.DISCRETE, [0, 1]],
        [ArrayDiscreteSpace(2, 0, 5), [0, 1], RLBaseTypes.CONTINUOUS, RLTypes.CONTINUOUS, [0, 1]],
        [ArrayDiscreteSpace(2, 0, 5), [0, 1], RLBaseTypes.ANY, RLTypes.DISCRETE, [0, 1]],
        [ContinuousSpace(0, 5), 1.2, RLBaseTypes.DISCRETE, RLTypes.DISCRETE, 1],
        [ContinuousSpace(0, 5), 1.2, RLBaseTypes.CONTINUOUS, RLTypes.CONTINUOUS, 1.2],
        [ContinuousSpace(0, 5), 1.2, RLBaseTypes.ANY, RLTypes.CONTINUOUS, 1.2],
        [ArrayContinuousSpace(1, 0, 5), [1.1, 2.1], RLBaseTypes.DISCRETE, RLTypes.DISCRETE, [1, 2]],
        [ArrayContinuousSpace(1, 0, 5), [1.1, 2.1], RLBaseTypes.CONTINUOUS, RLTypes.CONTINUOUS, [1.1, 2.1]],
        [ArrayContinuousSpace(1, 0, 5), [1.1, 2.1], RLBaseTypes.ANY, RLTypes.CONTINUOUS, [1.1, 2.1]],
        [BoxSpace((2, 1)), [[1.1], [2.1]], RLBaseTypes.DISCRETE, RLTypes.DISCRETE, [[1], [2]]],
        [BoxSpace((2, 1)), [[1.1], [2.1]], RLBaseTypes.CONTINUOUS, RLTypes.CONTINUOUS, [[1.1], [2.1]]],
        [BoxSpace((2, 1)), [[1.1], [2.1]], RLBaseTypes.ANY, RLTypes.CONTINUOUS, [[1.1], [2.1]]],
    ],
)
def test_observation_encode(env_obs_space, env_state, rl_obs_type, true_obs_type, true_state):
    print(env_obs_space, env_state, rl_obs_type, true_obs_type, true_state)
    env = srl.make_env("Stub")
    env_org = cast(StubEnv, env.unwrapped)
    env_org._observation_type = EnvObservationTypes.UNKNOWN
    env_org._observation_space = env_obs_space
    env_org.s_state = env_state

    rl_config = StubRLConfig()
    rl_config._observation_type = rl_obs_type

    # ---
    worker_run = srl.make_worker(rl_config, env)
    worker = cast(StubRLWorker, worker_run.worker)

    print(rl_config.observation_type)
    assert rl_config.observation_type == true_obs_type

    env.reset()
    worker_run.on_reset(0, training=False)
    action = worker_run.policy()

    print(worker.state)
    assert isinstance(worker.state, np.ndarray)
    assert np.allclose(worker.state, np.array(true_state))

    env.step(action)
    worker_run.on_step()

    print(worker.state)
    assert isinstance(worker.state, np.ndarray)
    assert np.allclose(worker.state, np.array(true_state))


@pytest.mark.parametrize(
    "env_obs_type",
    [
        EnvObservationTypes.GRAY_2ch,
        EnvObservationTypes.GRAY_3ch,
        EnvObservationTypes.COLOR,
        EnvObservationTypes.IMAGE,
    ],
)
@pytest.mark.parametrize(
    " env_obs_space, env_state, rl_obs_type, true_obs_type, true_state",
    [
        [BoxSpace((2, 1)), [[1.1], [2.1]], RLBaseTypes.DISCRETE, RLTypes.DISCRETE, [[1], [2]]],
        [BoxSpace((2, 1)), [[1.1], [2.1]], RLBaseTypes.CONTINUOUS, RLTypes.IMAGE, [[1.1], [2.1]]],
        [BoxSpace((2, 1)), [[1.1], [2.1]], RLBaseTypes.ANY, RLTypes.IMAGE, [[1.1], [2.1]]],
    ],
)
def test_observation_img_encode(env_obs_type, env_obs_space, env_state, rl_obs_type, true_obs_type, true_state):
    print(env_obs_type, env_obs_space, env_state, rl_obs_type, true_obs_type, true_state)
    env = srl.make_env("Stub")
    env_org = cast(StubEnv, env.unwrapped)
    env_org._observation_type = env_obs_type
    env_org._observation_space = env_obs_space
    env_org.s_state = env_state

    rl_config = StubRLConfig()
    rl_config._observation_type = rl_obs_type

    # ---
    worker_run = srl.make_worker(rl_config, env)
    worker = cast(StubRLWorker, worker_run.worker)

    print(rl_config.observation_type)
    assert rl_config.observation_type == true_obs_type

    env.reset()
    worker_run.on_reset(0, training=False)
    action = worker_run.policy()

    print(worker.state)
    assert isinstance(worker.state, np.ndarray)
    assert np.allclose(worker.state, np.array(true_state))

    env.step(action)
    worker_run.on_step()

    print(worker.state)
    assert isinstance(worker.state, np.ndarray)
    assert np.allclose(worker.state, np.array(true_state))


@pytest.mark.parametrize(
    "rl_action_type, env_action_space, true_type",
    [
        [RLBaseTypes.DISCRETE, DiscreteSpace(5), int],
        [RLBaseTypes.DISCRETE, ArrayDiscreteSpace(2, 0, 5), int],
        [RLBaseTypes.DISCRETE, ContinuousSpace(0, 5), int],
        [RLBaseTypes.DISCRETE, ArrayContinuousSpace(1), None],
        [RLBaseTypes.DISCRETE, ArrayContinuousSpace(1, 0, 5), int],
        [RLBaseTypes.DISCRETE, BoxSpace((1,)), None],
        [RLBaseTypes.DISCRETE, BoxSpace((1,), -1, 1), int],
        [RLBaseTypes.CONTINUOUS, DiscreteSpace(5), list],
        [RLBaseTypes.CONTINUOUS, ArrayDiscreteSpace(2, 0, 5), list],
        [RLBaseTypes.CONTINUOUS, ContinuousSpace(0, 5), list],
        [RLBaseTypes.CONTINUOUS, ArrayContinuousSpace(1), list],
        [RLBaseTypes.CONTINUOUS, ArrayContinuousSpace(1, 0, 5), list],
        [RLBaseTypes.CONTINUOUS, BoxSpace((1,)), list],
        [RLBaseTypes.CONTINUOUS, BoxSpace((1,), -1, 1), list],
    ],
)
def test_sample_action(rl_action_type, env_action_space, true_type):
    print(rl_action_type, env_action_space, true_type)
    env = srl.make_env("Stub")
    env_org = cast(StubEnv, env.unwrapped)
    env_org._action_space = env_action_space

    rl_config = StubRLConfig()
    rl_config._action_type = rl_action_type

    # ---
    if true_type is None:
        with pytest.raises(AssertionError):
            worker_run = srl.make_worker(rl_config, env)
        return
    else:
        worker_run = srl.make_worker(rl_config, env)

    action = worker_run.sample_action()
    print(action)
    print(worker_run.config.action_type)
    print(worker_run.config.action_space)
    if isinstance(true_type, list):
        assert isinstance(action, list)
        for a in action:
            assert isinstance(a, float)
    else:
        assert isinstance(action, true_type)


@pytest.mark.parametrize(
    "env_action_space",
    [
        DiscreteSpace(5),
        ArrayDiscreteSpace(2, 0, 5),
        ContinuousSpace(0, 5),
        ArrayContinuousSpace(1),
        ArrayContinuousSpace(1, 0, 5),
        BoxSpace((1,)),
        BoxSpace((1,), -1, 1),
    ],
)
def test_sample_action_for_env(env_action_space):
    print(env_action_space)
    env = srl.make_env("Stub")
    env_org = cast(StubEnv, env.unwrapped)
    env_org._action_space = env_action_space

    rl_config = StubRLConfig()

    # ---
    worker_run = srl.make_worker(rl_config, env)

    action = worker_run.sample_action_for_env()
    print(action)
    print(worker_run.config.action_type)
    print(worker_run.config.action_space)
    assert env_action_space.check_val(action)
