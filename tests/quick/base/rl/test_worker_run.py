from typing import Any, Tuple, cast

import numpy as np
import pytest

import srl
from srl.base.define import InfoType, RLActionType, RLBaseTypes, SpaceTypes
from srl.base.env.base import SpaceBase
from srl.base.env.genre.singleplay import SinglePlayEnv
from srl.base.env.registration import register as register_env
from srl.base.rl.config import RLConfig
from srl.base.rl.registration import register as register_rl
from srl.base.rl.worker import RLWorker
from srl.base.spaces import ArrayDiscreteSpace, BoxSpace, DiscreteSpace
from srl.base.spaces.array_continuous import ArrayContinuousSpace
from srl.base.spaces.continuous import ContinuousSpace
from srl.base.spaces.multi import MultiSpace
from srl.test.env import TestEnv
from srl.utils import common

_B_DIS = RLBaseTypes.DISCRETE
_B_CON = RLBaseTypes.CONTINUOUS
_B_ANY = RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS

_R_DIS = SpaceTypes.DISCRETE
_R_CON = SpaceTypes.CONTINUOUS
_R_IMG = SpaceTypes.IMAGE
_R_MUL = SpaceTypes.MULTI


class StubEnv(SinglePlayEnv):
    def __init__(self):
        self._action_space: SpaceBase = DiscreteSpace(5)
        self._observation_space: SpaceBase = DiscreteSpace(5)

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
        self._action_type = _B_ANY
        self._observation_type = _B_ANY

    def get_name(self) -> str:
        return "Stub"

    def get_base_action_type(self) -> RLBaseTypes:
        return self._action_type

    def get_base_observation_type(self) -> RLBaseTypes:
        return self._observation_type

    def get_framework(self) -> str:
        return ""


class StubRLWorker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.on_reset_state = np.array(0)
        self.state = np.array(0)
        self.action = 0

    def on_reset(self, worker) -> dict:
        self.on_reset_state = worker.state
        return {}

    def policy(self, worker) -> Tuple[RLActionType, dict]:
        self.state = worker.state
        return self.action, {}

    def on_step(self, worker) -> InfoType:
        self.state = worker.state
        return {}


register_rl(StubRLConfig(), "", "", "", __name__ + ":StubRLWorker")


def test_env_play():
    tester = TestEnv()
    tester.play_test("Stub")


@pytest.mark.parametrize(
    "env_act_space, env_act, rl_act_type, true_act_space, true_rl_act",
    [
        [DiscreteSpace(5), 1, _B_DIS, DiscreteSpace(5), 1],
        [DiscreteSpace(5), 1, _B_CON, ArrayContinuousSpace(1, 0, 4), [1.0]],
        [DiscreteSpace(5), 1, _B_ANY, DiscreteSpace(5), 1],
        [ArrayDiscreteSpace(2, 0, 5), [0, 1], _B_DIS, DiscreteSpace(36), 1],
        [ArrayDiscreteSpace(2, 0, 5), [0, 1], _B_CON, ArrayContinuousSpace(2, 0, 5), [0.0, 1.0]],
        [ArrayDiscreteSpace(2, 0, 5), [0, 1], _B_ANY, DiscreteSpace(36), 1],
        [ContinuousSpace(0, 5), 1.2, _B_DIS, DiscreteSpace(5), 1],
        [ContinuousSpace(0, 5), 1.2, _B_CON, ArrayContinuousSpace(1, 0, 5), [1.2]],
        [ContinuousSpace(0, 5), 1.2, _B_ANY, ArrayContinuousSpace(1, 0, 5), [1.2]],
        [ArrayContinuousSpace(2, 0, 5), [1.1, 2.1], _B_DIS, DiscreteSpace(25), 7],
        [ArrayContinuousSpace(2, 0, 5), [1.1, 2.1], _B_CON, ArrayContinuousSpace(2, 0, 5), [1.1, 2.1]],
        [ArrayContinuousSpace(2, 0, 5), [1.1, 2.1], _B_ANY, ArrayContinuousSpace(2, 0, 5), [1.1, 2.1]],
        [BoxSpace((2, 1), -1, 1, stype=SpaceTypes.DISCRETE), np.array([[0], [1]]), _B_DIS, DiscreteSpace(9), 5],
        [
            BoxSpace((2, 1), -1, 1, stype=SpaceTypes.DISCRETE),
            np.array([[0], [1]]),
            _B_CON,
            ArrayContinuousSpace(2, -1, 1),
            [0, 1],
        ],
        [BoxSpace((2, 1), -1, 1, stype=SpaceTypes.DISCRETE), np.array([[0], [1]]), _B_ANY, DiscreteSpace(9), 5],
        [
            BoxSpace((2, 1), -1, 1, stype=SpaceTypes.CONTINUOUS),
            np.array([[0.1], [1.1]]),
            _B_DIS,
            DiscreteSpace(25),
            14,
        ],
        [
            BoxSpace((2, 1), -1, 1, stype=SpaceTypes.CONTINUOUS),
            np.array([[0.1], [1.1]]),
            _B_CON,
            ArrayContinuousSpace(2, -1, 1),
            [0.1, 1.1],
        ],
        [
            BoxSpace((2, 1), -1, 1, stype=SpaceTypes.CONTINUOUS),
            np.array([[0.1], [1.1]]),
            _B_ANY,
            ArrayContinuousSpace(2, -1, 1),
            [0.1, 1.1],
        ],
        # [MultiSpace([DiscreteSpace(5)]), [1], _B_ANY, _R_MUL, [1]],
        # [
        #     MultiSpace(
        #         [
        #             DiscreteSpace(5),
        #             ArrayDiscreteSpace(2, 0, 5),
        #             ContinuousSpace(0, 5),
        #             ArrayContinuousSpace(1, 0, 5),
        #             BoxSpace((2, 1), -1, 1),
        #         ]
        #     ),
        #     [
        #         1,
        #         [0, 1],
        #         1.2,
        #         [1.1, 2.1],
        #         np.array([[0.1], [1.1]]),
        #     ],
        #     _B_CON,
        #     _R_MUL,
        #     [
        #         1,
        #         [0, 1],
        #         1.2,
        #         [1.1, 2.1],
        #         np.array([[0.1], [1.1]]),
        #     ],
        # ],
    ],
)
def test_action_encode(env_act_space, env_act, rl_act_type, true_act_space, true_rl_act):
    print(env_act_space, env_act, rl_act_type, true_act_space, true_rl_act)
    common.logger_print()
    env = srl.make_env("Stub")
    env_org = cast(StubEnv, env.unwrapped)
    env_org._action_space = env_act_space

    rl_config = StubRLConfig()
    rl_config._action_type = rl_act_type

    # ---
    worker_run = srl.make_worker(rl_config, env)
    assert rl_config.action_space == true_act_space

    rl_act = worker_run.action_encode(env_act)
    print(rl_act)
    if true_act_space.stype == _R_DIS:  # int
        assert isinstance(rl_act, int)
        assert rl_act == true_rl_act
    elif true_act_space.stype == _R_CON:  # list[float]
        assert isinstance(rl_act, list)
        for a in rl_act:
            assert isinstance(a, float)
        np.testing.assert_array_equal(rl_act, true_rl_act)
    elif true_act_space.stype == _R_IMG:  # numpy int8
        # まだ未実装
        assert isinstance(rl_act, np.ndarray)
        assert np.array_equal(rl_act, true_rl_act) and rl_act.dtype == true_rl_act.dtype
    elif true_act_space.stype == _R_MUL:  # list[space val]
        assert isinstance(rl_act, list)
        assert len(rl_act) == len(true_rl_act)
        for i in range(len(true_rl_act)):
            if isinstance(true_rl_act[i], np.ndarray):
                assert (rl_act[i] == true_rl_act[i]).all()
            else:
                assert rl_act[i] == true_rl_act[i]
    else:
        raise


@pytest.mark.parametrize(
    "rl_act_type, rl_act, env_act_space, true_env_act",
    [
        [_B_DIS, 1, DiscreteSpace(5), 1],
        [_B_DIS, 1, ArrayDiscreteSpace(2, 0, 5), [0, 1]],
        [_B_DIS, 1, ContinuousSpace(0, 5), 1.25],
        [_B_DIS, 1, ArrayContinuousSpace(1, 0, 5), [1.25]],
        [_B_DIS, 2, BoxSpace((1,)), None],
        [_B_DIS, 1, BoxSpace((1,), -1, 1, dtype=np.float16), np.array([-0.5], np.float16)],
        [_B_CON, 1.2, DiscreteSpace(5), 1],
        [_B_CON, 1.2, ArrayDiscreteSpace(2, 0, 5), [1, 1]],
        [_B_CON, 1.2, ContinuousSpace(0, 5), 1.2],
        [_B_CON, 1.2, ArrayContinuousSpace(1, 0, 5), [1.2]],
        [_B_CON, 1.2, BoxSpace((1,), dtype=np.float16), np.array([1.2], dtype=np.float16)],
        [_B_CON, [1.2, 2.2], DiscreteSpace(5), 1],
        [_B_CON, [1.2, 2.2], ArrayDiscreteSpace(2, 0, 5), [1, 2]],
        [_B_CON, [1.2, 2.2], ContinuousSpace(0, 5), 1.2],
        [_B_CON, [1.2, 2.2], ArrayContinuousSpace(1, 0, 5), [1.2, 2.2]],
        [_B_CON, [1.2, 2.2], BoxSpace((2, 1), dtype=np.float16), np.array([[1.2], [2.2]], dtype=np.float16)],
        [_B_ANY, 1, DiscreteSpace(5), 1],
        [_B_ANY, 1, ArrayDiscreteSpace(2, 0, 5), [0, 1]],
        [_B_ANY, 1.2, ContinuousSpace(0, 5), 1.2],
        [_B_ANY, [1.2, 2.2], ArrayContinuousSpace(1, 0, 5), [1.2, 2.2]],
        [_B_ANY, [1.2, 2.2], BoxSpace((2, 1), dtype=np.float16), np.array([[1.2], [2.2]], dtype=np.float16)],
        # [_B_DIS, [1], MultiSpace([DiscreteSpace(5)]), [1]],
        # [_B_CON, [1], MultiSpace([DiscreteSpace(5)]), [1]],
        # [_B_CON, [1, 2], MultiSpace([DiscreteSpace(5), DiscreteSpace(5)]), [1, 2]],
    ],
)
def test_action_decode(rl_act_type, rl_act, env_act_space, true_env_act):
    print(rl_act_type, rl_act, env_act_space, true_env_act)
    common.logger_print()
    env = srl.make_env("Stub")
    env_org = cast(StubEnv, env.unwrapped)
    env_org._action_space = env_act_space

    rl_config = StubRLConfig()
    rl_config._action_type = rl_act_type

    # ---
    if true_env_act is None:
        with pytest.raises(AssertionError):
            worker_run = srl.make_worker(rl_config, env)
        return
    else:
        worker_run = srl.make_worker(rl_config, env)

    env.reset()
    worker_run.on_reset(0, training=False)
    worker = cast(StubRLWorker, worker_run.worker)

    worker.action = rl_act
    env_action = worker_run.policy()
    print(env_action)
    if isinstance(env_act_space, DiscreteSpace):  # int
        assert isinstance(env_action, int)
        assert env_action == true_env_act
    elif isinstance(env_act_space, ArrayDiscreteSpace):  # list[int]
        assert isinstance(env_action, list)
        for a in env_action:
            assert isinstance(a, int)
        np.testing.assert_array_equal(env_action, true_env_act)
    elif isinstance(env_act_space, ContinuousSpace):  # float
        assert isinstance(env_action, float)
        assert np.allclose(env_action, true_env_act)
    elif isinstance(env_act_space, ArrayContinuousSpace):  # list[float]
        assert isinstance(env_action, list)
        for a in env_action:
            assert isinstance(a, float)
        assert np.allclose(env_action, true_env_act)
    elif isinstance(env_act_space, BoxSpace):  # np
        assert isinstance(env_action, np.ndarray)
        assert np.array_equal(env_action, true_env_act) and env_action.dtype == true_env_act.dtype
    elif isinstance(env_act_space, MultiSpace):  # list
        assert isinstance(env_action, list)
        assert env_action == true_env_act
    else:
        raise


def _test_observation_encode(env_obs_space, env_state, rl_obs_type, true_obs_space, true_state):
    print(env_obs_space, env_state, rl_obs_type, true_obs_space, true_state)
    common.logger_print()
    env = srl.make_env("Stub")
    env_org = cast(StubEnv, env.unwrapped)
    env_org._observation_space = env_obs_space
    env_org.s_state = env_state

    rl_config = StubRLConfig()
    rl_config._observation_type = rl_obs_type

    # ---
    worker_run = srl.make_worker(rl_config, env)
    worker = cast(StubRLWorker, worker_run.worker)

    print(rl_config.observation_space)
    assert rl_config.observation_space == true_obs_space

    env.reset()
    worker_run.on_reset(0, training=False)
    action = worker_run.policy()

    print(worker.state)
    if true_obs_space.stype == SpaceTypes.DISCRETE:  # list int
        assert isinstance(worker.state, list)
        assert isinstance(worker.state[0], int)
        assert worker.state == true_state
    elif true_obs_space.stype == SpaceTypes.CONTINUOUS:  # numpy float32
        assert isinstance(worker.state, np.ndarray)
        assert worker.state.dtype == np.float32
        assert np.allclose(worker.state, true_state)
    elif true_obs_space.stype == SpaceTypes.IMAGE:  # numpy float32
        assert isinstance(worker.state, np.ndarray)
        assert worker.state.dtype == np.float32
        assert np.allclose(worker.state, true_state)
    elif true_obs_space.stype == SpaceTypes.MULTI:  # list
        assert isinstance(worker.state, list)
        assert len(worker.state) == len(true_state)
        for i in range(len(true_state)):
            if isinstance(true_state[i], np.ndarray):
                assert (worker.state[i] == true_state[i]).all()
            else:
                assert worker.state[i] == true_state[i]

    env.step(action)
    worker_run.on_step()

    print(worker.state)
    if true_obs_space.stype == SpaceTypes.DISCRETE:  # list int
        assert isinstance(worker.state, list)
        assert isinstance(worker.state[0], int)
        assert worker.state == true_state
    elif true_obs_space.stype == SpaceTypes.CONTINUOUS:  # numpy float32
        assert isinstance(worker.state, np.ndarray)
        assert worker.state.dtype == np.float32
        assert np.allclose(worker.state, true_state)
    elif true_obs_space.stype == SpaceTypes.IMAGE:  # numpy float32
        assert isinstance(worker.state, np.ndarray)
        assert worker.state.dtype == np.float32
        assert np.allclose(worker.state, true_state)
    elif true_obs_space.stype == SpaceTypes.MULTI:  # list
        assert isinstance(worker.state, list)
        assert len(worker.state) == len(true_state)
        for i in range(len(true_state)):
            if isinstance(true_state[i], np.ndarray):
                assert (worker.state[i] == true_state[i]).all()
            else:
                assert worker.state[i] == true_state[i]


@pytest.mark.parametrize(
    "env_obs_space, env_state, rl_obs_type, true_obs_type, true_state",
    [
        [DiscreteSpace(5), 1, _B_DIS, ArrayDiscreteSpace(1, 0, 4), [1]],
        [DiscreteSpace(5), 1, _B_CON, BoxSpace((1,), 0, 4, stype=_R_CON), np.array([1], np.float32)],
        [DiscreteSpace(5), 1, _B_ANY, ArrayDiscreteSpace(1, 0, 4), [1]],
        [ArrayDiscreteSpace(2, 0, 5), [0, 1], _B_DIS, ArrayDiscreteSpace(2, 0, 5), [0, 1]],
        [
            ArrayDiscreteSpace(2, 0, 5),
            [0, 1],
            _B_CON,
            BoxSpace((2,), 0, 5, stype=_R_CON),
            np.array([0, 1], np.float32),
        ],
        [ArrayDiscreteSpace(2, 0, 5), [0, 1], _B_ANY, ArrayDiscreteSpace(2, 0, 5), [0, 1]],
        [ContinuousSpace(0, 5), 1.2, _B_DIS, ArrayDiscreteSpace(1, 0, 5), [1]],
        [ContinuousSpace(0, 5), 1.2, _B_CON, BoxSpace((1,), 0, 5, stype=_R_CON), np.array([1.2], np.float32)],
        [ContinuousSpace(0, 5), 1.2, _B_ANY, BoxSpace((1,), 0, 5, stype=_R_CON), np.array([1.2], np.float32)],
        [ArrayContinuousSpace(2, 0, 5), [1.1, 2.1], _B_DIS, ArrayDiscreteSpace(2, 0, 5), [1, 2]],
        [
            ArrayContinuousSpace(2, 0, 5),
            [1.1, 2.1],
            _B_CON,
            BoxSpace((2,), 0, 5, stype=_R_CON),
            np.array([1.1, 2.1], np.float32),
        ],
        [
            ArrayContinuousSpace(2, 0, 5),
            [1.1, 2.1],
            _B_ANY,
            BoxSpace((2,), 0, 5, stype=_R_CON),
            np.array([1.1, 2.1], np.float32),
        ],
        [BoxSpace((2, 1), -1, 5), [[1.1], [2.1]], _B_DIS, ArrayDiscreteSpace(2, -1, 5), [1, 2]],
        [
            BoxSpace((2, 1), -1, 5),
            [[1.1], [2.1]],
            _B_CON,
            BoxSpace((2, 1), -1, 5, stype=_R_CON),
            np.array([[1.1], [2.1]], np.float32),
        ],
        [
            BoxSpace((2, 1), -1, 5),
            [[1.1], [2.1]],
            _B_ANY,
            BoxSpace((2, 1), -1, 5, stype=_R_CON),
            np.array([[1.1], [2.1]], np.float32),
        ],
        # [MultiSpace([DiscreteSpace(5)]), [1], _B_ANY, ArrayDiscreteSpace(1, 0, 5), [1]],
        # [
        #     MultiSpace(
        #         [
        #             DiscreteSpace(5),
        #             ArrayDiscreteSpace(2, 0, 5),
        #             ContinuousSpace(0, 5),
        #             ArrayContinuousSpace(1, 0, 5),
        #             BoxSpace((2, 1)),
        #         ]
        #     ),
        #     [
        #         1,
        #         [0, 1],
        #         1.2,
        #         [1.1, 2.1],
        #         [[1.1], [2.1]],
        #     ],
        #     _B_CON,
        #     _R_MUL,
        #     [
        #         np.array([1], np.float32),
        #         np.array([0, 1], np.float32),
        #         np.array([1.2], np.float32),
        #         np.array([1.1, 2.1], np.float32),
        #         np.array([[1.1], [2.1]], np.float32),
        #     ],
        # ],
    ],
)
def test_observation_encode(env_obs_space, env_state, rl_obs_type, true_obs_type, true_state):
    _test_observation_encode(env_obs_space, env_state, rl_obs_type, true_obs_type, true_state)


@pytest.mark.parametrize(
    "env_obs_space, env_state, rl_obs_type, true_obs_type, true_state",
    [
        [
            BoxSpace((2, 1), -1, 5, stype=SpaceTypes.GRAY_2ch),
            [[1.1], [2.1]],
            _B_DIS,
            ArrayDiscreteSpace(2, -1, 5),
            [1, 2],
        ],
        [
            BoxSpace((2, 1), -1, 5, stype=SpaceTypes.GRAY_3ch),
            [[1.1], [2.1]],
            _B_DIS,
            ArrayDiscreteSpace(2, -1, 5),
            [1, 2],
        ],
        [
            BoxSpace((2, 1), -1, 5, stype=SpaceTypes.COLOR),
            [[1.1], [2.1]],
            _B_DIS,
            ArrayDiscreteSpace(2, -1, 5),
            [1, 2],
        ],
        [
            BoxSpace((2, 1), -1, 5, stype=SpaceTypes.IMAGE),
            [[1.1], [2.1]],
            _B_DIS,
            ArrayDiscreteSpace(2, -1, 5),
            [1, 2],
        ],
        [
            BoxSpace((2, 1), -1, 5, stype=SpaceTypes.GRAY_2ch),
            [[1.1], [2.1]],
            _B_CON,
            BoxSpace((2, 1), -1, 5, stype=SpaceTypes.CONTINUOUS),
            [[1.1], [2.1]],
        ],
        [
            BoxSpace((2, 1), -1, 5, stype=SpaceTypes.GRAY_3ch),
            [[1.1], [2.1]],
            _B_CON,
            BoxSpace((2, 1), -1, 5, stype=SpaceTypes.CONTINUOUS),
            [[1.1], [2.1]],
        ],
        [
            BoxSpace((2, 1), -1, 5, stype=SpaceTypes.COLOR),
            [[1.1], [2.1]],
            _B_CON,
            BoxSpace((2, 1), -1, 5, stype=SpaceTypes.CONTINUOUS),
            [[1.1], [2.1]],
        ],
        [
            BoxSpace((2, 1), -1, 5, stype=SpaceTypes.IMAGE),
            [[1.1], [2.1]],
            _B_CON,
            BoxSpace((2, 1), -1, 5, stype=SpaceTypes.CONTINUOUS),
            [[1.1], [2.1]],
        ],
        [
            BoxSpace((2, 1), -1, 5, stype=SpaceTypes.GRAY_2ch),
            [[1.1], [2.1]],
            RLBaseTypes.IMAGE,
            BoxSpace((2, 1), -1, 5, stype=SpaceTypes.GRAY_2ch),
            [[1.1], [2.1]],
        ],
        [
            BoxSpace((2, 1), -1, 5, stype=SpaceTypes.GRAY_3ch),
            [[1.1], [2.1]],
            RLBaseTypes.IMAGE,
            BoxSpace((2, 1), -1, 5, stype=SpaceTypes.GRAY_3ch),
            [[1.1], [2.1]],
        ],
        [
            BoxSpace((2, 1), -1, 5, stype=SpaceTypes.COLOR),
            [[1.1], [2.1]],
            RLBaseTypes.IMAGE,
            BoxSpace((2, 1), -1, 5, stype=SpaceTypes.COLOR),
            [[1.1], [2.1]],
        ],
        [
            BoxSpace((2, 1), -1, 5, stype=SpaceTypes.IMAGE),
            [[1.1], [2.1]],
            RLBaseTypes.IMAGE,
            BoxSpace((2, 1), -1, 5, stype=SpaceTypes.IMAGE),
            [[1.1], [2.1]],
        ],
    ],
)
def test_observation_img_encode(env_obs_space, env_state, rl_obs_type, true_obs_type, true_state):
    _test_observation_encode(env_obs_space, env_state, rl_obs_type, true_obs_type, true_state)


@pytest.mark.parametrize(
    "rl_action_type, env_action_space, true_type",
    [
        [_B_DIS, DiscreteSpace(5), int],
        [_B_DIS, ArrayDiscreteSpace(2, 0, 5), int],
        [_B_DIS, ContinuousSpace(0, 5), int],
        [_B_DIS, ArrayContinuousSpace(1), None],
        [_B_DIS, ArrayContinuousSpace(1, 0, 5), int],
        [_B_DIS, BoxSpace((1,)), None],
        [_B_DIS, BoxSpace((1,), -1, 1), int],
        [_B_CON, DiscreteSpace(5), list],
        [_B_CON, ArrayDiscreteSpace(2, 0, 5), list],
        [_B_CON, ContinuousSpace(0, 5), list],
        [_B_CON, ArrayContinuousSpace(1), list],
        [_B_CON, ArrayContinuousSpace(1, 0, 5), list],
        [_B_CON, BoxSpace((1,)), list],
        [_B_CON, BoxSpace((1,), -1, 1), list],
    ],
)
def test_sample_action(rl_action_type, env_action_space, true_type):
    print(rl_action_type, env_action_space, true_type)
    common.logger_print()
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
    print(worker_run.config.action_space)
    assert env_action_space.check_val(action)
