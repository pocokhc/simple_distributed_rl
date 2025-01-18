from typing import Any, Optional, cast

import numpy as np
import pytest

import srl
from srl.base.context import RunContext
from srl.base.define import ObservationModes, RLActionType, RLBaseActTypes, RLBaseObsTypes, SpaceTypes
from srl.base.env.base import EnvBase
from srl.base.env.registration import register as register_env
from srl.base.rl.config import RLConfig
from srl.base.rl.registration import register as register_rl
from srl.base.rl.worker import RLWorker
from srl.base.spaces.array_continuous import ArrayContinuousSpace
from srl.base.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.continuous import ContinuousSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.multi import MultiSpace
from srl.base.spaces.space import SpaceBase
from srl.test.env import env_test
from srl.utils import common

_D = DiscreteSpace
_AD = ArrayDiscreteSpace
_C = ContinuousSpace
_AC = ArrayContinuousSpace
_B = BoxSpace
_M = MultiSpace


class StubEnv(EnvBase):
    def __init__(self, action_space=DiscreteSpace(7), observation_space=DiscreteSpace(7)):
        self._action_space = action_space
        self._observation_space = observation_space

        self.s_state: Any = 0
        self.s_reward = 0.0
        self.s_done = True
        self.s_action = 0

    @property
    def action_space(self) -> SpaceBase:
        return self._action_space

    @property
    def observation_space(self) -> SpaceBase:
        return self._observation_space

    @property
    def player_num(self) -> int:
        return 1

    @property
    def max_episode_steps(self) -> int:
        return 0

    def reset(self, **kwargs):
        return self.s_state

    def step(self, action):
        self.s_action = action
        return self.s_state, self.s_reward, self.s_done, False

    def backup(self, **kwargs) -> Any:
        return None

    def restore(self, state: Any, **kwargs) -> None:
        pass  # do nothing

    def render_terminal(self):
        print("a")

    def render_rgb_array(self, **kwargs) -> np.ndarray:
        return np.ones((64, 32, 3))


class StubRLConfig(RLConfig):
    def __init__(self) -> None:
        super().__init__()
        self._action_type = RLBaseActTypes.DISCRETE
        self._observation_type = RLBaseObsTypes.DISCRETE
        self._use_render_image_state = False

    def get_name(self) -> str:
        return "Stub"

    def get_base_action_type(self) -> RLBaseActTypes:
        return self._action_type

    def get_base_observation_type(self) -> RLBaseObsTypes:
        return self._observation_type

    def get_framework(self) -> str:
        return ""

    def use_render_image_state(self) -> bool:
        return self._use_render_image_state


class StubRLWorker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.on_reset_state = np.array(0)
        self.state = np.array(0)
        self.action = 0

    def on_reset(self, worker):
        self.on_reset_state = worker.state

    def policy(self, worker) -> RLActionType:
        self.state = worker.state
        return self.action

    def on_step(self, worker):
        self.state = worker.state


@pytest.fixture(scope="function", autouse=True)
def scope_function():
    register_env(id="Stub", entry_point=__name__ + ":StubEnv", check_duplicate=False)
    register_rl(StubRLConfig(), "", "", "", __name__ + ":StubRLWorker", check_duplicate=False)
    yield


def test_env_play():
    env_test("Stub")


# ------------------------------------------------------------------


def _test_action(
    env_act_space: SpaceBase,
    rl_act_type: RLBaseActTypes,
    rl_act_type_override: RLBaseActTypes,
    true_act_space: SpaceBase,
    rl_act,
    env_act,
):
    common.logger_print()
    env = srl.make_env(srl.EnvConfig("Stub", {"action_space": env_act_space}))

    rl_config = StubRLConfig()
    rl_config.enable_assertion = True
    rl_config._action_type = rl_act_type
    rl_config.override_action_type = rl_act_type_override

    worker_run = srl.make_worker(rl_config, env)
    worker = cast(StubRLWorker, worker_run.worker)

    context = RunContext()
    env.setup()
    worker_run.on_start(context)

    # --- check act space
    print(rl_config.action_space)
    print(true_act_space)
    assert rl_config.action_space == true_act_space

    # --- encode
    enc_rl_act = worker_run.action_encode(env_act)
    print(enc_rl_act)
    print(rl_act)
    assert rl_config.action_space.check_val(enc_rl_act)
    if isinstance(enc_rl_act, np.ndarray):
        assert (rl_act == enc_rl_act).all()
    else:
        assert rl_act == enc_rl_act

    # --- decode
    env.reset()
    worker_run.on_reset(0)
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
        [_C(0, 1), 10, 2, 0.2222222222222222],
        [_AC(2, -1, 1), 9, 0, [-1.0, -1.0]],
        [_B((2, 1), -1, 1), 9, 0, np.array([[-1], [-1]], np.float32)],
    ],
)
def test_action_discrete(env_act_space, n, rl_act, env_act):
    # int, DiscreteSpace
    _test_action(
        env_act_space,
        rl_act_type=RLBaseActTypes.DISCRETE,
        rl_act_type_override=RLBaseActTypes.NONE,
        true_act_space=DiscreteSpace(n),
        rl_act=rl_act,
        env_act=env_act,
    )


@pytest.mark.parametrize(
    "env_act_space, true_space_args, rl_act, env_act",
    [
        [_D(5), [1, 0, 4], [2.0], 2],
        [_AD(2, -1, 1), [2, -1, 1], [1.0, 1.0], [1, 1]],
        [_C(0, 1), [1, 0, 1], [1.0], 1.0],
        [_AC(2, -1, 1), [2, -1, 1], [1.0, 1.0], [1.0, 1.0]],
        [_B((2, 1), -1, 1), [2, -1, 1], [1.0, 1.0], np.array([[1.0], [1.0]], np.float32)],
    ],
)
def test_action_continuous(env_act_space, true_space_args, rl_act, env_act):
    # list[float] ArrayContinuousSpace
    _test_action(
        env_act_space,
        rl_act_type=RLBaseActTypes.CONTINUOUS,
        rl_act_type_override=RLBaseActTypes.NONE,
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
        rl_act_type=RLBaseActTypes.NONE,
        rl_act_type_override=RLBaseActTypes.NONE,
        true_act_space=BoxSpace(*true_space_args),
        rl_act=rl_act,
        env_act=env_act,
    )


@pytest.mark.parametrize(
    "env_act_space, true_act_space, rl_act, env_act",
    [
        [_D(5), _D(5), 2, 2],
        [_AD(2, -1, 1), _D(3 * 3), 2, [-1, 1]],
        [_C(0, 1), _AC(1, 0, 1), [1.0], 1.0],
        [_AC(2, -1, 1), _AC(2, -1, 1), [-1.0, -0.5], [-1.0, -0.5]],
        [_B((2, 1), -1, 1), _AC(2, -1, 1), [-1.0, -0.5], np.array([[-1], [-0.5]], np.float32)],
    ],
)
def test_action_disc_cont(env_act_space, true_act_space, rl_act, env_act):
    _test_action(
        env_act_space,
        rl_act_type=RLBaseActTypes.DISCRETE | RLBaseActTypes.CONTINUOUS,
        rl_act_type_override=RLBaseActTypes.NONE,
        true_act_space=true_act_space,
        rl_act=rl_act,
        env_act=env_act,
    )


# ---------------------------------------------------------------


def _test_obs(
    env_obs_space: SpaceBase,
    rl_obs_type: RLBaseObsTypes,
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
    use_render_image_state=False,
    render_image_window_length=1,
):
    if rl_obs_mode == ObservationModes.RENDER_IMAGE:
        pytest.importorskip("PIL")
        pytest.importorskip("pygame")

    common.logger_print()
    env = srl.make_env(srl.EnvConfig("Stub", {"observation_space": env_obs_space}))
    env_org = cast(StubEnv, env.unwrapped)
    env_org.s_state = env_state

    rl_config = StubRLConfig()
    rl_config.enable_assertion = True
    rl_config._observation_type = rl_obs_type
    rl_config.window_length = window_length
    rl_config.observation_mode = rl_obs_mode
    rl_config.override_observation_type = rl_obs_type_override
    rl_config.observation_division_num = rl_obs_div_num
    rl_config._use_render_image_state = use_render_image_state
    rl_config.render_image_window_length = render_image_window_length

    worker_run = srl.make_worker(rl_config, env)
    worker = cast(StubRLWorker, worker_run.worker)

    context = RunContext()
    env.setup()
    worker_run.on_start(context)

    # --- check obs space
    print(true_obs_space)
    print(rl_config.observation_space)
    print(rl_config.observation_space_of_env)
    assert rl_config.observation_space == true_obs_space
    assert rl_config.observation_space_of_env == true_obs_env_space
    if window_length == 1:
        assert rl_config.observation_space_one_step == rl_config.observation_space
    else:
        assert true_obs_space_one_step is not None
        assert rl_config.observation_space_one_step == true_obs_space_one_step

    # --- check val
    env.reset()
    worker_run.on_reset(0)
    env_action = worker_run.policy()

    if use_render_image_state:
        true_img_space = BoxSpace((64, 32, 3), 0, 255, np.uint8, stype=SpaceTypes.COLOR)
        assert rl_config.obs_render_img_space_one_step == true_img_space
        if render_image_window_length == 1:
            assert (np.ones((64, 32, 3)) == worker_run.render_img_state).all()
            assert rl_config.obs_render_img_space == true_img_space
        else:
            true_img_state = np.stack(
                [
                    np.zeros((64, 32, 3)),
                    np.zeros((64, 32, 3)),
                    np.zeros((64, 32, 3)),
                    np.ones((64, 32, 3)),
                ],
                axis=0,
            )
            assert (true_img_state == worker_run.render_img_state).all()
            assert rl_config.obs_render_img_space == BoxSpace((4, 64, 32, 3), 0, 255, np.uint8, stype=SpaceTypes.COLOR)

    print(worker_run.state)
    print(true_state1)
    assert true_obs_space.check_val(worker_run.state)
    if isinstance(true_obs_space, MultiSpace):
        for i in range(true_obs_space.space_size):
            assert (np.array(worker_run.state[i]) == np.array(true_state1[i])).all()
    elif isinstance(worker_run.state, np.ndarray):
        assert (worker_run.state == true_state1).all()
    else:
        assert worker_run.state == true_state1

    env.step(env_action)
    worker_run.on_step()

    if use_render_image_state:
        true_img_space = BoxSpace((64, 32, 3), 0, 255, np.uint8, stype=SpaceTypes.COLOR)
        assert rl_config.obs_render_img_space_one_step == true_img_space
        if render_image_window_length == 1:
            assert (np.ones((64, 32, 3)) == worker_run.render_img_state).all()
            assert rl_config.obs_render_img_space == true_img_space
        else:
            true_img_state = np.stack(
                [
                    np.zeros((64, 32, 3)),
                    np.zeros((64, 32, 3)),
                    np.ones((64, 32, 3)),
                    np.ones((64, 32, 3)),
                ],
                axis=0,
            )
            assert (true_img_state == worker_run.render_img_state).all()
            assert rl_config.obs_render_img_space == BoxSpace((4, 64, 32, 3), 0, 255, np.uint8, stype=SpaceTypes.COLOR)

    print(worker_run.state)
    print(true_state2)
    assert true_obs_space.check_val(worker_run.state)
    if isinstance(true_obs_space, MultiSpace):
        for i in range(true_obs_space.space_size):
            assert (np.array(worker_run.state[i]) == np.array(true_state2[i])).all()
    elif isinstance(worker_run.state, np.ndarray):
        assert (worker_run.state == true_state2).all()
    else:
        assert worker_run.state == true_state2


@pytest.mark.parametrize(
    "env_obs_space, rl_obs_div_num, true_space_args, env_state, true_state",
    [
        [DiscreteSpace(5), -1, [1, 0, 4], 1, [1]],
        [ArrayDiscreteSpace(2, 0, 5), -1, [2, 0, 5], [0, 1], [0, 1]],
        [ContinuousSpace(0, 5), -1, [1, 0, 5], 1.2, [1]],
        [ContinuousSpace(0, 5), 6, [1, 0, 6], 1.2, [1]],
        [ArrayContinuousSpace(2, 0, 5), -1, [2, 0, 5], [1.1, 2.1], [1, 2]],
        [ArrayContinuousSpace(2, 0, 5), 6, [1, 0, 4], [1.1, 2.1], [0]],
        [BoxSpace((2, 1), -1, 5), -1, [2, -1, 5], [[1.1], [2.1]], [1, 2]],
        [BoxSpace((2, 1), 0, 5, dtype=np.uint8), -1, [2, 0, 5], [[1], [2]], [1, 2]],
        [BoxSpace((2, 1), 0, 5), 6, [1, 0, 4], [[1.1], [2.1]], [0]],
    ],
)
def test_obs_discrete(env_obs_space, rl_obs_div_num, env_state, true_space_args, true_state):
    # list[int], ArrayDiscreteSpace
    _test_obs(
        env_obs_space=env_obs_space,
        rl_obs_type=RLBaseObsTypes.DISCRETE,
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
        rl_obs_type=RLBaseObsTypes.DISCRETE,
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
        [_D(5), [(1,), 0, 4, np.float32, SpaceTypes.DISCRETE], 1, [1]],
        [_AD(2, 0, 5), [(2,), 0, 5, np.float32, SpaceTypes.DISCRETE], [0, 1], [0, 1]],
        [_C(0, 5), [(1,), 0, 5], 1.2, [1.2]],
        [_AC(2, 0, 5), [(2,), 0, 5], [1.1, 2.1], [1.1, 2.1]],
        [_B((2, 1), -1, 5), [(2, 1), -1, 5], [[1.1], [2.1]], [[1.1], [2.1]]],
        [_B((2, 1), 0, 5, np.uint8), [(2, 1), 0, 5, np.float32, SpaceTypes.DISCRETE], [[1], [2]], [[1], [2]]],
    ],
)
def test_obs_box(env_obs_space, env_state, true_space_args, true_state):
    # NDArray[np.float32] BoxSpace
    true_state = np.array(true_state, np.float32)
    _test_obs(
        env_obs_space=env_obs_space,
        rl_obs_type=RLBaseObsTypes.BOX,
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
        [
            _D(5),
            [(1,), 0, 4, np.float32, SpaceTypes.DISCRETE],
            [(2, 1), 0, 4, np.float32, SpaceTypes.DISCRETE],
            1,
            [[0], [1]],
            [[1], [1]],
        ],
        [
            _AD(2, 0, 5),
            [(2,), 0, 5, np.float32, SpaceTypes.DISCRETE],
            [(2, 2), 0, 5, np.float32, SpaceTypes.DISCRETE],
            [0, 1],
            [[0, 0], [0, 1]],
            [[0, 1], [0, 1]],
        ],
        [_C(0, 5), [(1,), 0, 5], [(2, 1), 0, 5], 1.2, [[0], [1.2]], [[1.2], [1.2]]],
        [_AC(2, 0, 5), [(2,), 0, 5], [(2, 2), 0, 5], [1.1, 2.1], [[0.0, 0.0], [1.1, 2.1]], [[1.1, 2.1], [1.1, 2.1]]],
        [_B((1, 1), -1, 5), [(1, 1), -1, 5], [(2, 1, 1), -1, 5], [[1.1]], [[[0.0]], [[1.1]]], [[[1.1]], [[1.1]]]],
    ],
)
def test_obs_box_window(
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
        rl_obs_type=RLBaseObsTypes.BOX,
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
        rl_obs_type=RLBaseObsTypes.BOX,
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
        env_obs_space=BoxSpace((64, 32), stype=SpaceTypes.GRAY_2ch),
        rl_obs_type=RLBaseObsTypes.BOX,
        rl_obs_mode=ObservationModes.ENV,
        rl_obs_type_override=SpaceTypes.UNKNOWN,
        rl_obs_div_num=-1,
        true_obs_space=BoxSpace((64, 32, 2), stype=SpaceTypes.IMAGE),
        true_obs_env_space=BoxSpace((64, 32), stype=SpaceTypes.GRAY_2ch),
        window_length=2,
        true_obs_space_one_step=BoxSpace((64, 32), stype=SpaceTypes.GRAY_2ch),
        env_state=np.zeros((64, 32)),
        true_state1=np.zeros((64, 32, 2)),
        true_state2=np.zeros((64, 32, 2)),
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
        rl_obs_type=RLBaseObsTypes.BOX,
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


# ---------------------------------------------------------


def test_obs_render_image():
    _test_obs(
        env_obs_space=DiscreteSpace(1),
        rl_obs_type=RLBaseObsTypes.NONE,
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


def test_obs_render_terminal():
    pytest.skip("TODO")
    _test_obs(
        env_obs_space=DiscreteSpace(1),
        rl_obs_type=RLBaseObsTypes.NONE,
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


@pytest.mark.parametrize(
    "rl_obs_mode, window_length, render_image_window_length, true_obs_env_space, true_obs_space, env_state, true_state1, true_state2",
    [
        [ObservationModes.ENV, 1, 1, DiscreteSpace(2), DiscreteSpace(2), 1, 1, 1],
        [ObservationModes.ENV, 1, 4, DiscreteSpace(2), DiscreteSpace(2), 1, 1, 1],
        [ObservationModes.ENV, 4, 1, DiscreteSpace(2), ArrayDiscreteSpace(4, 0, 1), 1, [0, 0, 0, 1], [0, 0, 1, 1]],
        [ObservationModes.ENV, 4, 4, DiscreteSpace(2), ArrayDiscreteSpace(4, 0, 1), 1, [0, 0, 0, 1], [0, 0, 1, 1]],
        [
            ObservationModes.RENDER_IMAGE,
            4,
            1,
            BoxSpace((64, 32, 3), 0, 255, np.uint8, stype=SpaceTypes.COLOR),
            BoxSpace((4, 64, 32, 3), 0, 255, np.uint8, stype=SpaceTypes.COLOR),
            np.ones((64, 32, 3)),
            np.stack(
                [
                    np.zeros((64, 32, 3)),
                    np.zeros((64, 32, 3)),
                    np.zeros((64, 32, 3)),
                    np.ones((64, 32, 3)),
                ],
                axis=0,
            ),
            np.stack(
                [
                    np.zeros((64, 32, 3)),
                    np.zeros((64, 32, 3)),
                    np.ones((64, 32, 3)),
                    np.ones((64, 32, 3)),
                ],
                axis=0,
            ),
        ],
    ],
)
def test_obs_use_render_img_state(
    rl_obs_mode,
    window_length,
    render_image_window_length,
    true_obs_env_space,
    true_obs_space,
    env_state,
    true_state1,
    true_state2,
):
    pytest.importorskip("pygame")
    _test_obs(
        env_obs_space=DiscreteSpace(2),
        rl_obs_type=RLBaseObsTypes.NONE,
        rl_obs_mode=rl_obs_mode,
        rl_obs_type_override=SpaceTypes.UNKNOWN,
        rl_obs_div_num=-1,
        true_obs_space=true_obs_space,
        true_obs_env_space=true_obs_env_space,
        window_length=window_length,
        true_obs_space_one_step=true_obs_env_space,
        env_state=env_state,
        true_state1=true_state1,
        true_state2=true_state2,
        use_render_image_state=True,
        render_image_window_length=render_image_window_length,
    )


# ---------------------------------------------------------


def test_obs_override():
    _test_obs(
        env_obs_space=BoxSpace((64, 64), stype=SpaceTypes.CONTINUOUS),
        rl_obs_type=RLBaseObsTypes.NONE,
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
        RLBaseActTypes.NONE,
        RLBaseActTypes.DISCRETE,
        RLBaseActTypes.CONTINUOUS,
        RLBaseActTypes.DISCRETE | RLBaseActTypes.CONTINUOUS,
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
    ],
)
def test_sample_action(env_act_space, rl_act_type):
    if env_act_space.is_image() and (rl_act_type & RLBaseActTypes.DISCRETE):
        pytest.skip("intに変換できない")
    common.logger_print()
    env = srl.make_env(srl.EnvConfig("Stub", {"action_space": env_act_space}))

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
    env = srl.make_env(srl.EnvConfig("Stub", {"action_space": env_act_space}))
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
