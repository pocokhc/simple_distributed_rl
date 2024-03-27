from typing import Any, Optional, Tuple, cast

import numpy as np
import pytest

import srl
from srl.base.define import InfoType, ObservationModes, RLActionType, RLBaseTypes, SpaceTypes
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

_D = DiscreteSpace
_AD = ArrayDiscreteSpace
_C = ContinuousSpace
_AC = ArrayContinuousSpace
_B = BoxSpace
_M = MultiSpace


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

    def render_terminal(self):
        print("a")

    def render_rgb_array(self, **kwargs) -> np.ndarray:
        return np.ones((64, 32, 3))


register_env(
    id="Stub",
    entry_point=__name__ + ":StubEnv",
)


class StubRLConfig(RLConfig):
    def __init__(self) -> None:
        super().__init__()
        self._action_type = RLBaseTypes.DISCRETE
        self._observation_type = RLBaseTypes.DISCRETE

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


# ------------------------------------------------------------------


def _test_action(
    env_act_space: SpaceBase,
    rl_act_type: RLBaseTypes,
    rl_act_type_override: SpaceTypes,
    true_act_space: SpaceBase,
    rl_act,
    env_act,
):
    common.logger_print()
    env = srl.make_env("Stub")
    env_org = cast(StubEnv, env.unwrapped)
    env_org._action_space = env_act_space

    rl_config = StubRLConfig()
    rl_config.enable_assertion = True
    rl_config._action_type = rl_act_type
    rl_config.override_action_type = rl_act_type_override

    worker_run = srl.make_worker(rl_config, env)
    worker = cast(StubRLWorker, worker_run.worker)

    # --- check act space
    print(rl_config.action_space)
    print(true_act_space)
    assert rl_config.action_space == true_act_space

    # --- encode
    enc_rl_act = worker_run.action_encode(env_act)
    print(enc_rl_act)
    assert rl_config.action_space.check_val(enc_rl_act)
    if isinstance(enc_rl_act, np.ndarray):
        assert (rl_act == enc_rl_act).all()
    else:
        assert rl_act == enc_rl_act

    # --- decode
    env.reset()
    worker_run.on_reset(0, training=False)
    worker.action = rl_act
    dec_env_action = worker_run.policy()
    print(dec_env_action)
    assert env_act_space.check_val(dec_env_action)
    if isinstance(dec_env_action, np.ndarray):
        assert (dec_env_action == env_act).all()
    else:
        assert dec_env_action == env_act


@pytest.mark.parametrize(
    "env_act_space, n, rl_act, env_act",
    [
        [_D(5), 5, 2, 2],
        [_AD(2, -1, 1), 3 * 3, 2, [-1, 1]],
        [_C(0, 1), 5, 1, 0.25],
        [_AC(2, -1, 1), 5 * 5, 1, [-1.0, -0.5]],
        [_B((2, 1), -1, 1), 5 * 5, 1, np.array([[-1], [-0.5]], np.float32)],
        [
            MultiSpace(
                [
                    DiscreteSpace(2),
                    ArrayDiscreteSpace(1, 0, 1),
                    ContinuousSpace(0, 1),
                    ArrayContinuousSpace(1, 0, 1),
                    BoxSpace((1, 1), 0, 1),
                ]
            ),
            2 * 2 * 5 * 5 * 5,
            2,
            [0, [0], 0.0, [0.0], np.array([[0.5]], np.float32)],
        ],
    ],
)
def test_action_discrete(env_act_space, n, rl_act, env_act):
    # int, DiscreteSpace
    _test_action(
        env_act_space,
        rl_act_type=RLBaseTypes.DISCRETE,
        rl_act_type_override=SpaceTypes.UNKNOWN,
        true_act_space=DiscreteSpace(n),
        rl_act=rl_act,
        env_act=env_act,
    )


@pytest.mark.parametrize(
    "env_act_space, true_space_args, env_act, rl_act",
    [
        [_D(5), [1, 0, 4], 2, [2.0]],
        [_AD(2, -1, 1), [2, -1, 1], [1, 1], [1.0, 1.0]],
        [_C(0, 1), [1, 0, 1], 1.0, [1.0]],
        [_AC(2, -1, 1), [2, -1, 1], [1.0, 1.0], [1.0, 1.0]],
        [_B((2, 1), -1, 1), [2, -1, 1], np.array([[1.0], [1.0]], np.float32), [1.0, 1.0]],
        [
            MultiSpace(
                [
                    DiscreteSpace(3),
                    ArrayDiscreteSpace(1, 0, 1),
                    ContinuousSpace(0, 1),
                    ArrayContinuousSpace(1, 0, 1),
                    BoxSpace((1, 1), 0, 1),
                ]
            ),
            [5, 0, [2, 1, 1, 1, 1]],
            [0, [0], 0.0, [0.0], np.array([[0.5]], np.float32)],
            [0.0, 0.0, 0.0, 0.0, 0.5],
        ],
    ],
)
def test_action_continuous(env_act_space, true_space_args, env_act, rl_act):
    # list[float] ArrayContinuousSpace
    _test_action(
        env_act_space,
        rl_act_type=RLBaseTypes.CONTINUOUS,
        rl_act_type_override=SpaceTypes.UNKNOWN,
        true_act_space=ArrayContinuousSpace(*true_space_args),
        rl_act=rl_act,
        env_act=env_act,
    )


@pytest.mark.parametrize(
    "env_act_space, true_space_args, env_act, rl_act",
    [
        [
            _B((64, 64), 0, 1, stype=SpaceTypes.GRAY_3ch),
            [(64, 64), 0, 1, np.float32, SpaceTypes.GRAY_3ch],
            np.zeros((64, 64), np.float32),
            np.zeros((64, 64), np.float32),
        ],
    ],
)
def test_action_image(env_act_space, true_space_args, env_act, rl_act):
    # NDArray[np.uint8] BoxSpace
    _test_action(
        env_act_space,
        rl_act_type=RLBaseTypes.IMAGE,
        rl_act_type_override=SpaceTypes.UNKNOWN,
        true_act_space=BoxSpace(*true_space_args),
        rl_act=rl_act,
        env_act=env_act,
    )


@pytest.mark.parametrize(
    "env_act_space, rl_act_space, is_multi, env_act, rl_act",
    [
        [_D(5), _D(5), False, 2, [2]],
        [_AD(2, -1, 1), _D(9), False, [1, 1], [8]],
        [_C(0, 1), _AC(1, 0, 1), False, 1.0, [[1.0]]],
        [_AC(2, -1, 1), _AC(2, -1, 1), False, [1.0, 1.0], [[1.0, 1.0]]],
        [_B((2, 1), -1, 1), _AC(2, -1, 1), False, np.array([[1.0], [1.0]], np.float32), [[1.0, 1.0]]],
        [
            MultiSpace(
                [
                    DiscreteSpace(3),
                    ArrayDiscreteSpace(1, 0, 1),
                    ContinuousSpace(0, 1),
                    ArrayContinuousSpace(1, 0, 1),
                    BoxSpace((1, 1), 0, 1),
                ]
            ),
            MultiSpace(
                [
                    DiscreteSpace(3),
                    DiscreteSpace(2),
                    ArrayContinuousSpace(1, 0, 1),
                    ArrayContinuousSpace(1, 0, 1),
                    ArrayContinuousSpace(1, 0, 1),
                ]
            ),
            True,
            [0, [0], 0.0, [0.0], np.array([[0.5]], np.float32)],
            [0, 0, [0.0], [0.0], [0.5]],
        ],
    ],
)
def test_action_multi(env_act_space, rl_act_space, is_multi, env_act, rl_act):
    if is_multi:
        true_act_space = rl_act_space
    else:
        true_act_space = MultiSpace([rl_act_space])
    _test_action(
        env_act_space,
        rl_act_type=RLBaseTypes.MULTI,
        rl_act_type_override=SpaceTypes.UNKNOWN,
        true_act_space=true_act_space,
        rl_act=rl_act,
        env_act=env_act,
    )


# ---------------------------------------------------------------


def _test_obs(
    env_obs_space: SpaceBase,
    rl_obs_type: RLBaseTypes,
    rl_obs_mode: ObservationModes,
    rl_obs_type_override: SpaceTypes,
    rl_obs_div_num: int,
    true_obs_space: SpaceBase,
    true_obs_env_space: SpaceBase,
    window_length: int,
    true_obs_space_one_step: Optional[SpaceBase],
    env_state,
    true_state1,
    true_state2,
):
    if rl_obs_mode == ObservationModes.RENDER_IMAGE:
        pytest.importorskip("PIL")
        pytest.importorskip("pygame")

    common.logger_print()
    env = srl.make_env("Stub")
    env_org = cast(StubEnv, env.unwrapped)
    env_org._observation_space = env_obs_space
    env_org.s_state = env_state

    rl_config = StubRLConfig()
    rl_config.enable_assertion = True
    rl_config._observation_type = rl_obs_type
    rl_config.window_length = window_length
    rl_config.observation_mode = rl_obs_mode
    rl_config.override_observation_type = rl_obs_type_override
    rl_config.observation_division_num = rl_obs_div_num

    worker_run = srl.make_worker(rl_config, env)
    worker = cast(StubRLWorker, worker_run.worker)

    # --- check obs space
    print(rl_config.observation_space)
    print(true_obs_space)
    assert rl_config.observation_space == true_obs_space
    assert rl_config.observation_space_of_env == true_obs_env_space
    if window_length == 1:
        assert rl_config.observation_space_one_step == rl_config.observation_space
    else:
        assert true_obs_space_one_step is not None
        assert rl_config.observation_space_one_step == true_obs_space_one_step

    # --- check val
    env.reset()
    worker_run.on_reset(0, training=False)
    env_action = worker_run.policy()

    print(worker.state)
    print(true_state1)
    assert true_obs_space.check_val(worker.state)
    if isinstance(true_obs_space, MultiSpace):
        for i in range(true_obs_space.space_size):
            assert (np.array(worker.state[i]) == np.array(true_state1[i])).all()
    elif isinstance(worker.state, np.ndarray):
        assert (worker.state == true_state1).all()
    else:
        assert worker.state == true_state1

    env.step(env_action)
    worker_run.on_step()

    print(worker.state)
    print(true_state2)
    assert true_obs_space.check_val(worker.state)
    if isinstance(true_obs_space, MultiSpace):
        for i in range(true_obs_space.space_size):
            assert (np.array(worker.state[i]) == np.array(true_state2[i])).all()
    elif isinstance(worker.state, np.ndarray):
        assert (worker.state == true_state2).all()
    else:
        assert worker.state == true_state2


@pytest.mark.parametrize(
    "env_obs_space, rl_obs_div_num, true_space_args, env_state, true_state",
    [
        [DiscreteSpace(5), -1, [1, 0, 4], 1, [1]],
        [ArrayDiscreteSpace(2, 0, 5), -1, [2, 0, 5], [0, 1], [0, 1]],
        [ContinuousSpace(0, 5), -1, [1, 0, 5], 1.2, [1]],
        [ContinuousSpace(0, 5), 6, [1, 0, 6], 1.2, [1]],
        [ArrayContinuousSpace(2, 0, 5), -1, [2, 0, 5], [1.1, 2.1], [1, 2]],
        [ArrayContinuousSpace(2, 0, 5), 6, [1, 0, 6 * 6], [1.1, 2.1], [8]],
        [BoxSpace((2, 1), -1, 5), -1, [2, -1, 5], [[1.1], [2.1]], [1, 2]],
        [BoxSpace((2, 1), 0, 5, dtype=np.uint8), -1, [2, 0, 5], [[1], [2]], [1, 2]],
        [BoxSpace((2, 1), 0, 5), 6, [1, 0, 6 * 6], [[1.1], [2.1]], [8]],
        [
            MultiSpace(
                [
                    DiscreteSpace(5),
                    ArrayDiscreteSpace(2, 0, 5),
                    ContinuousSpace(0, 2),
                    ArrayContinuousSpace(2, 0, 2),
                    BoxSpace((2, 1), 0, 2),
                ]
            ),
            3,
            [6, [0, 0, 0, 0, 0, 0], [4, 5, 5, 3, 9, 9]],
            [1, [1, 1], 1.1, [1.1, 1.1], np.array([[1], [1]])],
            [1, 1, 1, 1, 4, 4],
        ],
    ],
)
def test_obs_discrete(env_obs_space, rl_obs_div_num, env_state, true_space_args, true_state):
    # list[int], ArrayDiscreteSpace
    _test_obs(
        env_obs_space=env_obs_space,
        rl_obs_type=RLBaseTypes.DISCRETE,
        rl_obs_mode=ObservationModes.ENV,
        rl_obs_type_override=SpaceTypes.UNKNOWN,
        rl_obs_div_num=rl_obs_div_num,
        true_obs_space=ArrayDiscreteSpace(*true_space_args),
        true_obs_env_space=env_obs_space,
        window_length=1,
        true_obs_space_one_step=None,
        env_state=env_state,
        true_state1=true_state,
        true_state2=true_state,
    )


@pytest.mark.parametrize(
    "env_obs_space, true_one_space_args, true_space_args, env_state, true_state1, true_state2",
    [
        [_D(5), [1, 0, 4], [2, 0, 4], 1, [0, 1], [1, 1]],
        [_AD(2, 0, 5), [2, 0, 5], [4, 0, 5], [0, 1], [0, 0, 0, 1], [0, 1, 0, 1]],
        [_C(0, 5), [1, 0, 5], [2, 0, 5], 1.2, [0, 1], [1, 1]],
        [_AC(2, 0, 5), [2, 0, 5], [4, 0, 5], [1.1, 2.1], [0, 0, 1, 2], [1, 2, 1, 2]],
        [_B((2, 1), -1, 5), [2, -1, 5], [4, -1, 5], [[1.1], [2.1]], [0, 0, 1, 2], [1, 2, 1, 2]],
        [
            MultiSpace(
                [
                    DiscreteSpace(5),
                    ArrayDiscreteSpace(1, 0, 5),
                    ContinuousSpace(0, 2),
                    ArrayContinuousSpace(1, 0, 2),
                    BoxSpace((1, 1), 0, 2),
                ]
            ),
            [5, [0, 0, 0, 0, 0], [4, 5, 2, 2, 2]],
            [10, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [4, 5, 2, 2, 2, 4, 5, 2, 2, 2]],
            [3, [1], 1.1, [1.1], np.array([[1]])],
            [0, 0, 0, 0, 0, 3, 1, 1, 1, 1],
            [3, 1, 1, 1, 1, 3, 1, 1, 1, 1],
        ],
    ],
)
def test_obs_discrete_window(
    env_obs_space,
    true_one_space_args,
    true_space_args,
    env_state,
    true_state1,
    true_state2,
):
    # list[int], ArrayDiscreteSpace
    _test_obs(
        env_obs_space=env_obs_space,
        rl_obs_type=RLBaseTypes.DISCRETE,
        rl_obs_mode=ObservationModes.ENV,
        rl_obs_type_override=SpaceTypes.UNKNOWN,
        rl_obs_div_num=-1,
        true_obs_space=ArrayDiscreteSpace(*true_space_args),
        true_obs_env_space=env_obs_space,
        window_length=2,
        true_obs_space_one_step=ArrayDiscreteSpace(*true_one_space_args),
        env_state=env_state,
        true_state1=true_state1,
        true_state2=true_state2,
    )


@pytest.mark.parametrize(
    "env_obs_space, true_space_args, env_state, true_state",
    [
        [_D(5), [(1,), 0, 4], 1, [1]],
        [_AD(2, 0, 5), [(2,), 0, 5], [0, 1], [0, 1]],
        [_C(0, 5), [(1,), 0, 5], 1.2, [1.2]],
        [_AC(2, 0, 5), [(2,), 0, 5], [1.1, 2.1], [1.1, 2.1]],
        [_B((2, 1), -1, 5), [(2, 1), -1, 5], [[1.1], [2.1]], [[1.1], [2.1]]],
        [_B((2, 1), 0, 5, np.uint8), [(2, 1), 0, 5, np.float32], [[1], [2]], [[1], [2]]],
        [
            MultiSpace(
                [
                    DiscreteSpace(5),
                    ArrayDiscreteSpace(2, 0, 5),
                    ContinuousSpace(0, 2),
                    ArrayContinuousSpace(2, 0, 2),
                    BoxSpace((2, 1), 0, 2),
                ]
            ),
            [(1 + 2 + 1 + 2 + 2,), 0, [4.0, 5.0, 5.0, 2.0, 2.0, 2.0, 2.0, 2.0]],
            [2, [1, 1], 1.1, [1.1, 1.1], np.array([[1], [1]])],
            [2, 1, 1, 1.1, 1.1, 1.1, 1, 1],
        ],
    ],
)
def test_obs_continuous(env_obs_space, env_state, true_space_args, true_state):
    # NDArray[np.float32] BoxSpace
    true_state = np.array(true_state, np.float32)
    _test_obs(
        env_obs_space=env_obs_space,
        rl_obs_type=RLBaseTypes.CONTINUOUS,
        rl_obs_mode=ObservationModes.ENV,
        rl_obs_type_override=SpaceTypes.UNKNOWN,
        rl_obs_div_num=-1,
        true_obs_space=BoxSpace(*true_space_args),
        true_obs_env_space=env_obs_space,
        window_length=1,
        true_obs_space_one_step=None,
        env_state=env_state,
        true_state1=true_state,
        true_state2=true_state,
    )


@pytest.mark.parametrize(
    "env_obs_space, true_one_space_args, true_space_args, env_state, true_state1, true_state2",
    [
        [_D(5), [(1,), 0, 4], [(2, 1), 0, 4], 1, [[0], [1]], [[1], [1]]],
        [_AD(2, 0, 5), [(2,), 0, 5], [(2, 2), 0, 5], [0, 1], [[0, 0], [0, 1]], [[0, 1], [0, 1]]],
        [_C(0, 5), [(1,), 0, 5], [(2, 1), 0, 5], 1.2, [[0], [1.2]], [[1.2], [1.2]]],
        [_AC(2, 0, 5), [(2,), 0, 5], [(2, 2), 0, 5], [1.1, 2.1], [[0.0, 0.0], [1.1, 2.1]], [[1.1, 2.1], [1.1, 2.1]]],
        [_B((1, 1), -1, 5), [(1, 1), -1, 5], [(2, 1, 1), -1, 5], [[1.1]], [[[0.0]], [[1.1]]], [[[1.1]], [[1.1]]]],
        [
            MultiSpace(
                [
                    DiscreteSpace(5),
                    ArrayDiscreteSpace(1, 0, 5),
                    ContinuousSpace(0, 2),
                    ArrayContinuousSpace(1, 0, 2),
                    BoxSpace((1, 1), 0, 2),
                ]
            ),
            [(1 + 1 + 1 + 1 + 1,), 0, [4.0, 5.0, 2.0, 2.0, 2.0]],
            [(2, 1 + 1 + 1 + 1 + 1), 0, 5],
            [2, [1], 1.1, [1.1], np.array([[1]])],
            [[0, 0, 0, 0, 0], [2, 1, 1.1, 1.1, 1]],
            [[2, 1, 1.1, 1.1, 1], [2, 1, 1.1, 1.1, 1]],
        ],
    ],
)
def test_obs_continuous_window(
    env_obs_space,
    true_one_space_args,
    true_space_args,
    env_state,
    true_state1,
    true_state2,
):
    # NDArray[np.float32] BoxSpace
    true_state1 = np.array(true_state1, np.float32)
    true_state2 = np.array(true_state2, np.float32)
    _test_obs(
        env_obs_space=env_obs_space,
        rl_obs_type=RLBaseTypes.CONTINUOUS,
        rl_obs_mode=ObservationModes.ENV,
        rl_obs_type_override=SpaceTypes.UNKNOWN,
        rl_obs_div_num=-1,
        true_obs_space=BoxSpace(*true_space_args),
        true_obs_env_space=env_obs_space,
        window_length=2,
        true_obs_space_one_step=BoxSpace(*true_one_space_args),
        env_state=env_state,
        true_state1=true_state1,
        true_state2=true_state2,
    )


@pytest.mark.parametrize("env_stype", [SpaceTypes.GRAY_2ch, SpaceTypes.GRAY_3ch, SpaceTypes.COLOR, SpaceTypes.IMAGE])
def test_obs_image(env_stype):
    # NDArray[np.float32] BoxSpace
    _test_obs(
        env_obs_space=BoxSpace((64,), stype=env_stype),
        rl_obs_type=RLBaseTypes.IMAGE,
        rl_obs_mode=ObservationModes.ENV,
        rl_obs_type_override=SpaceTypes.UNKNOWN,
        rl_obs_div_num=-1,
        true_obs_space=BoxSpace((64,), stype=env_stype),
        true_obs_env_space=BoxSpace((64,), stype=env_stype),
        window_length=1,
        true_obs_space_one_step=None,
        env_state=np.zeros((64,)),
        true_state1=np.zeros((64,)),
        true_state2=np.zeros((64,)),
    )


def test_obs_image_window_gray_2ch():
    _test_obs(
        env_obs_space=BoxSpace((64,), stype=SpaceTypes.GRAY_2ch),
        rl_obs_type=RLBaseTypes.IMAGE,
        rl_obs_mode=ObservationModes.ENV,
        rl_obs_type_override=SpaceTypes.UNKNOWN,
        rl_obs_div_num=-1,
        true_obs_space=BoxSpace((64, 2), stype=SpaceTypes.IMAGE),
        true_obs_env_space=BoxSpace((64,), stype=SpaceTypes.GRAY_2ch),
        window_length=2,
        true_obs_space_one_step=BoxSpace((64,), stype=SpaceTypes.GRAY_2ch),
        env_state=np.zeros((64,)),
        true_state1=np.zeros((64, 2)),
        true_state2=np.zeros((64, 2)),
    )


@pytest.mark.parametrize(
    "env_stype",
    [
        SpaceTypes.GRAY_3ch,
        SpaceTypes.COLOR,
        SpaceTypes.IMAGE,
    ],
)
def test_obs_image_window(env_stype):
    _test_obs(
        env_obs_space=BoxSpace((64,), stype=env_stype),
        rl_obs_type=RLBaseTypes.IMAGE,
        rl_obs_mode=ObservationModes.ENV,
        rl_obs_type_override=SpaceTypes.UNKNOWN,
        rl_obs_div_num=-1,
        true_obs_space=BoxSpace((2, 64), stype=env_stype),
        true_obs_env_space=BoxSpace((64,), stype=env_stype),
        window_length=2,
        true_obs_space_one_step=BoxSpace((64,), stype=env_stype),
        env_state=np.zeros((64,)),
        true_state1=np.zeros((2, 64)),
        true_state2=np.zeros((2, 64)),
    )


@pytest.mark.parametrize(
    "env_obs_space, rl_obs_space, is_multi, env_state, true_rl_state",
    [
        [_D(5), _AD(1, 0, 4), False, 1, [[1]]],
        [_AD(2, 0, 5), _AD(2, 0, 5), False, [0, 1], [[0, 1]]],
        [_C(0, 5), _B((1,), 0, 5), False, 1.2, [np.array([1.2], np.float32)]],
        [_AC(2, 0, 5), _B((2,), 0, 5), False, [1.1, 2.1], [np.array([1.1, 2.1], np.float32)]],
        [
            _B((2, 1), -1, 5),
            _B((2, 1), -1, 5),
            False,
            np.array([[1.1], [2.1]]),
            [np.array([[1.1], [2.1]], np.float32)],
        ],
        [
            MultiSpace(
                [
                    DiscreteSpace(5),
                    ArrayDiscreteSpace(2, 0, 5),
                    ContinuousSpace(0, 2),
                    ArrayContinuousSpace(2, 0, 2),
                    BoxSpace((2, 1), 0, 2),
                ]
            ),
            MultiSpace(
                [
                    ArrayDiscreteSpace(1, 0, 4),
                    ArrayDiscreteSpace(2, 0, 5),
                    BoxSpace((1,), 0, 2),
                    BoxSpace((2,), 0, 2),
                    BoxSpace((2, 1), 0, 2),
                ]
            ),
            True,
            [2, [1, 1], 1.1, [1.1, 1.1], np.array([[1], [1]])],
            [
                [2],
                [1, 1],
                np.array([1.1], np.float32),
                np.array([1.1, 1.1], np.float32),
                np.array([[1], [1]], np.float32),
            ],
        ],
    ],
)
def test_obs_multi(env_obs_space, rl_obs_space, is_multi, env_state, true_rl_state):
    if is_multi:
        true_obs_space = rl_obs_space
    else:
        true_obs_space = MultiSpace([rl_obs_space])
    _test_obs(
        env_obs_space=env_obs_space,
        rl_obs_type=RLBaseTypes.MULTI,
        rl_obs_mode=ObservationModes.ENV,
        rl_obs_type_override=SpaceTypes.UNKNOWN,
        rl_obs_div_num=-1,
        true_obs_space=true_obs_space,
        true_obs_env_space=env_obs_space,
        window_length=1,
        true_obs_space_one_step=None,
        env_state=env_state,
        true_state1=true_rl_state,
        true_state2=true_rl_state,
    )


@pytest.mark.parametrize(
    "env_obs_space, rl_one_obs_space, rl_obs_space, is_multi, env_state, true_rl_state1, true_rl_state2",
    [
        [_D(5), _AD(1, 0, 4), _AD(2, 0, 4), False, 1, [[[0, 1]]], [[[1, 1]]]],
        [_AD(2, 0, 5), _AD(2, 0, 5), _AD(4, 0, 5), False, [0, 1], [[0, 0, 0, 1]], [[0, 1, 0, 1]]],
        [
            _C(0, 5),
            _B((1,), 0, 5),
            _B((2, 1), 0, 5),
            False,
            1.2,
            [np.array([[0], [1.2]], np.float32)],
            [np.array([[1.2], [1.2]], np.float32)],
        ],
        [
            _AC(2, 0, 5),
            _B((2,), 0, 5),
            _B((2, 2), 0, 5),
            False,
            [1.1, 2.1],
            [np.array([[0, 0], [1.1, 2.1]], np.float32)],
            [np.array([[1.1, 2.1], [1.1, 2.1]], np.float32)],
        ],
        [
            _B((2, 1), -1, 5),
            _B((2, 1), -1, 5),
            _B((2, 2, 1), -1, 5),
            False,
            np.array([[1.1], [2.1]]),
            [np.array([[[0], [0]], [[1.1], [2.1]]], np.float32)],
            [np.array([[[1.1], [2.1]], [[1.1], [2.1]]], np.float32)],
        ],
        [
            MultiSpace(
                [
                    DiscreteSpace(5),
                    ArrayDiscreteSpace(2, 0, 5),
                    ContinuousSpace(0, 2),
                    ArrayContinuousSpace(2, 0, 2),
                    BoxSpace((2, 1), 0, 2),
                ]
            ),
            MultiSpace(
                [
                    ArrayDiscreteSpace(1, 0, 4),
                    ArrayDiscreteSpace(2, 0, 5),
                    BoxSpace((1,), 0, 2),
                    BoxSpace((2,), 0, 2),
                    BoxSpace((2, 1), 0, 2),
                ]
            ),
            MultiSpace(
                [
                    ArrayDiscreteSpace(1 * 2, 0, 4),
                    ArrayDiscreteSpace(2 * 2, 0, 5),
                    BoxSpace((2, 1), 0, 2),
                    BoxSpace((2, 2), 0, 2),
                    BoxSpace((2, 2, 1), 0, 2),
                ]
            ),
            True,
            [2, [1, 1], 1.1, [1.1, 1.1], np.array([[1], [1]])],
            [
                [0, 2],
                [0, 0, 1, 1],
                np.array([[0], [1.1]], np.float32),
                np.array([[0, 0], [1.1, 1.1]], np.float32),
                np.array([[[0], [0]], [[1], [1]]], np.float32),
            ],
            [
                [2, 2],
                [1, 1, 1, 1],
                np.array([[1.1], [1.1]], np.float32),
                np.array([[1.1, 1.1], [1.1, 1.1]], np.float32),
                np.array([[[1], [1]], [[1], [1]]], np.float32),
            ],
        ],
    ],
)
def test_obs_multi_window(
    env_obs_space,
    rl_one_obs_space,
    rl_obs_space,
    is_multi,
    env_state,
    true_rl_state1,
    true_rl_state2,
):
    if is_multi:
        true_obs_space = rl_obs_space
        true_obs_space_one_step = rl_one_obs_space
    else:
        true_obs_space = MultiSpace([rl_obs_space])
        true_obs_space_one_step = MultiSpace([rl_one_obs_space])
    _test_obs(
        env_obs_space=env_obs_space,
        rl_obs_type=RLBaseTypes.MULTI,
        rl_obs_mode=ObservationModes.ENV,
        rl_obs_type_override=SpaceTypes.UNKNOWN,
        rl_obs_div_num=-1,
        true_obs_space=true_obs_space,
        true_obs_env_space=env_obs_space,
        window_length=2,
        true_obs_space_one_step=true_obs_space_one_step,
        env_state=env_state,
        true_state1=true_rl_state1,
        true_state2=true_rl_state2,
    )


# ---------------------------------------------------------


def test_obs_render_image():
    _test_obs(
        env_obs_space=DiscreteSpace(1),
        rl_obs_type=RLBaseTypes.NONE,
        rl_obs_mode=ObservationModes.RENDER_IMAGE,
        rl_obs_type_override=SpaceTypes.UNKNOWN,
        rl_obs_div_num=-1,
        true_obs_space=BoxSpace((64, 32, 3), 0, 255, np.uint8, stype=SpaceTypes.COLOR),
        true_obs_env_space=BoxSpace((64, 32, 3), 0, 255, np.uint8, stype=SpaceTypes.COLOR),
        window_length=1,
        true_obs_space_one_step=None,
        env_state=np.zeros((64, 32, 3)),
        true_state1=np.ones((64, 32, 3)),
        true_state2=np.ones((64, 32, 3)),
    )


def test_obs_env_render_image():
    true_space = MultiSpace(
        [
            ArrayDiscreteSpace(3, 0, 5),
            BoxSpace((64, 32, 3), 0, 255, np.uint8, stype=SpaceTypes.COLOR),
        ]
    )
    _test_obs(
        env_obs_space=ArrayDiscreteSpace(3, 0, 5),
        rl_obs_type=RLBaseTypes.NONE,
        rl_obs_mode=ObservationModes.ENV | ObservationModes.RENDER_IMAGE,
        rl_obs_type_override=SpaceTypes.UNKNOWN,
        rl_obs_div_num=-1,
        true_obs_space=true_space,
        true_obs_env_space=true_space,
        window_length=1,
        true_obs_space_one_step=None,
        env_state=[0, 1, 2],
        true_state1=[[0, 1, 2], np.ones((64, 32, 3))],
        true_state2=[[0, 1, 2], np.ones((64, 32, 3))],
    )


def test_obs_render_terminal():
    pytest.skip("TODO")
    _test_obs(
        env_obs_space=DiscreteSpace(1),
        rl_obs_type=RLBaseTypes.NONE,
        rl_obs_mode=ObservationModes.RENDER_TERMINAL,
        rl_obs_type_override=SpaceTypes.UNKNOWN,
        rl_obs_div_num=-1,
        true_obs_space=BoxSpace((64, 32, 3), 0, 255, np.uint8, stype=SpaceTypes.COLOR),
        true_obs_env_space=BoxSpace((64, 32, 3), 0, 255, np.uint8, stype=SpaceTypes.COLOR),
        window_length=1,
        true_obs_space_one_step=None,
        env_state=np.zeros((64, 32, 3)),
        true_state1=np.ones((64, 32, 3)),
        true_state2=np.ones((64, 32, 3)),
    )


# ---------------------------------------------------------


def test_obs_override():
    _test_obs(
        env_obs_space=BoxSpace((64, 64), stype=SpaceTypes.CONTINUOUS),
        rl_obs_type=RLBaseTypes.NONE,
        rl_obs_mode=ObservationModes.ENV,
        rl_obs_type_override=SpaceTypes.GRAY_3ch,
        rl_obs_div_num=-1,
        true_obs_space=BoxSpace((64, 64), stype=SpaceTypes.GRAY_3ch),
        true_obs_env_space=BoxSpace((64, 64), stype=SpaceTypes.GRAY_3ch),
        window_length=1,
        true_obs_space_one_step=None,
        env_state=np.zeros((64, 64)),
        true_state1=np.zeros((64, 64)),
        true_state2=np.zeros((64, 64)),
    )


# ---------------------------------------------------------


@pytest.mark.parametrize(
    "rl_act_type",
    [
        RLBaseTypes.DISCRETE,
        RLBaseTypes.CONTINUOUS,
        # RLBaseTypes.IMAGE,
        RLBaseTypes.MULTI,
        RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS,
        RLBaseTypes.CONTINUOUS | RLBaseTypes.MULTI,
        RLBaseTypes.DISCRETE | RLBaseTypes.CONTINUOUS | RLBaseTypes.MULTI,
    ],
)
@pytest.mark.parametrize(
    "env_act_space",
    [
        _D(5),
        _AD(2, -1, 1),
        _C(0, 1),
        _AC(2, -1, 1),
        _B((2, 1), -1, 1),
        _B((2, 1), -1, 1, np.int8),
        _B((2, 1), -1, 1, stype=SpaceTypes.COLOR),
        _M(
            [
                DiscreteSpace(2),
                ArrayDiscreteSpace(1, 0, 1),
                ContinuousSpace(0, 1),
                ArrayContinuousSpace(1, 0, 1),
                BoxSpace((1, 1), 0, 1),
            ]
        ),
    ],
)
def test_sample_action(env_act_space, rl_act_type):
    common.logger_print()
    env = srl.make_env("Stub")
    env_org = cast(StubEnv, env.unwrapped)
    env_org._action_space = env_act_space

    rl_config = StubRLConfig()
    rl_config.enable_assertion = True
    rl_config._action_type = rl_act_type

    worker_run = srl.make_worker(rl_config, env)

    print(worker_run.config.action_space)
    for _ in range(100):
        action = worker_run.sample_action()
        print(action)
        assert rl_config.action_space.check_val(action)


@pytest.mark.parametrize(
    "env_act_space, is_raise",
    [
        [DiscreteSpace(5), False],
        [ArrayDiscreteSpace(2, 0, 5), False],
        [ContinuousSpace(0, 5), False],
        [ArrayContinuousSpace(1), True],
        [ArrayContinuousSpace(1, 0, 5), False],
        [BoxSpace((1,)), True],
        [BoxSpace((1,), -1, 1), False],
    ],
)
def test_sample_action_for_env(env_act_space, is_raise):
    env = srl.make_env("Stub")
    env_org = cast(StubEnv, env.unwrapped)
    env_org._action_space = env_act_space

    rl_config = StubRLConfig()

    if is_raise:  # rangeの定義がない場合actionを定義できない
        with pytest.raises(AssertionError):
            worker_run = srl.make_worker(rl_config, env)
        return
    worker_run = srl.make_worker(rl_config, env)

    print(worker_run.config.action_space)
    for _ in range(100):
        action = worker_run.sample_action_for_env()
        print(action)
        assert env_act_space.check_val(action)
