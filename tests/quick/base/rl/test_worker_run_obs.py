from typing import Any, Literal, cast

import numpy as np
import pytest

import srl
from srl.base.context import RunContext
from srl.base.define import RLActionType, RLBaseActTypes, RLBaseObsTypes, SpaceTypes
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

        self.s_states: list = [1] * 10
        self.s_reward = 0.0
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
        return 5

    def reset(self, **kwargs):
        self.num_step = 0
        return self.s_states[self.num_step]

    def step(self, action):
        self.s_action = action
        self.num_step += 1
        self.s_reward += 1
        done = self.num_step == len(self.s_states) - 1
        return self.s_states[self.num_step], self.s_reward, done, False

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

    def use_render_image_state(self) -> bool:
        return self._use_render_image_state


class StubRLWorker(RLWorker):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.on_reset_state = np.array(0)
        self.state = np.array(0)
        self.action = 0
        self.tracking_size = 0

    def on_setup(self, worker, context: RunContext) -> None:
        if self.tracking_size > 0:
            worker.enable_tracking(self.tracking_size)

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
def _equal_data(a, b):
    if isinstance(a, np.ndarray):
        return (a == b).all()
    else:
        return a == b


def _test_obs_episode(
    env_obs_space: SpaceBase,
    rl_obs_type: RLBaseObsTypes,
    rl_obs_mode: Literal["", "render_image"],
    rl_obs_type_override: SpaceTypes,
    rl_obs_div_num: int,
    window_length: int,
    render_image_window_length,
    #
    true_obs_env_space: SpaceBase,
    true_obs_space_one: SpaceBase,
    true_obs_space: SpaceBase,
    #
    env_states: list,
    true_rl_states: list,
):
    if rl_obs_mode == "render_image":
        pytest.importorskip("PIL")
        pytest.importorskip("pygame")

    env = srl.make_env(srl.EnvConfig("Stub", {"observation_space": env_obs_space}))
    env_org = cast(StubEnv, env.unwrapped)
    env_org.s_states = env_states
    assert len(env_states) == 3

    rl_config = StubRLConfig()
    rl_config.enable_assertion = True
    rl_config._observation_type = rl_obs_type
    rl_config.window_length = window_length
    rl_config.observation_mode = rl_obs_mode
    rl_config.override_observation_type = rl_obs_type_override
    rl_config.observation_division_num = rl_obs_div_num
    rl_config._use_render_image_state = True
    rl_config.render_image_window_length = render_image_window_length

    # --- check rl_config setup
    rl_config.setup(env)
    print(true_obs_space)
    print(rl_config.observation_space)
    print(rl_config.observation_space_of_env)
    assert rl_config.observation_space_of_env == true_obs_env_space
    assert rl_config.observation_space_one_step == true_obs_space_one
    if window_length == 1:
        assert rl_config.observation_space == true_obs_space_one
    else:
        assert rl_config.observation_space == true_obs_space
    assert rl_config.used_rgb_array == True  # noqa: E712

    # render_image
    true_render_img_space_one = BoxSpace((64, 32, 3), 0, 255, np.uint8, stype=SpaceTypes.COLOR)
    if render_image_window_length == 1:
        true_render_img_space = true_render_img_space_one
    else:
        true_render_img_space = BoxSpace((render_image_window_length, 64, 32, 3), 0, 255, np.uint8, stype=SpaceTypes.COLOR)
    assert rl_config.obs_render_img_space_one_step == true_render_img_space_one
    assert rl_config.obs_render_img_space == true_render_img_space

    # --- setup
    worker = srl.make_worker(rl_config, env)

    context = RunContext()
    env.setup(context)
    worker.setup(context)

    # --- reset
    env.reset()
    worker.reset(0)
    env_action = worker.policy()
    if window_length == 1:
        assert _equal_data(worker.prev_state, true_obs_space_one.get_default())
        assert _equal_data(worker.state, true_rl_states[0])
    else:
        assert _equal_data(
            worker.prev_state,
            true_obs_space_one.encode_stack(
                [
                    true_obs_space_one.get_default(),
                    true_obs_space_one.get_default(),
                ]
            ),
        )
        assert _equal_data(
            worker.state,
            true_obs_space_one.encode_stack(
                [
                    true_obs_space_one.get_default(),
                    true_rl_states[0],
                ]
            ),
        )
    assert _equal_data(worker.state_one_step, true_rl_states[0])

    # render_image
    if render_image_window_length == 1:
        assert (worker.prev_render_image_state == true_render_img_space.get_default()).all()
        assert (worker.render_image_state == np.ones((64, 32, 3))).all()
    else:
        assert (worker.prev_render_image_state == np.zeros((2, 64, 32, 3))).all()
        state = np.stack(
            [
                np.zeros((64, 32, 3)),
                np.ones((64, 32, 3)),
            ],
            axis=0,
        )
        assert (worker.render_image_state == state).all()
    assert (worker.render_image_state_one_step == np.ones((64, 32, 3))).all()

    # --- step 1
    env.step(env_action)
    worker.on_step()
    env_action = worker.policy()
    if window_length == 1:
        assert _equal_data(worker.prev_state, true_rl_states[0])
        assert _equal_data(worker.state, true_rl_states[1])
    else:
        assert _equal_data(
            worker.prev_state,
            true_obs_space_one.encode_stack(
                [
                    true_obs_space_one.get_default(),
                    true_rl_states[0],
                ]
            ),
        )
        assert _equal_data(
            worker.state,
            true_obs_space_one.encode_stack(
                [
                    true_rl_states[0],
                    true_rl_states[1],
                ]
            ),
        )
    assert _equal_data(worker.state_one_step, true_rl_states[1])

    # render_image
    if render_image_window_length == 1:
        assert (worker.prev_render_image_state == np.ones((64, 32, 3))).all()
        assert (worker.render_image_state == np.ones((64, 32, 3))).all()
    else:
        state = np.stack(
            [
                np.zeros((64, 32, 3)),
                np.ones((64, 32, 3)),
            ],
            axis=0,
        )
        assert (worker.prev_render_image_state == state).all()
        assert (worker.render_image_state == np.ones((2, 64, 32, 3))).all()
    assert (worker.render_image_state_one_step == np.ones((64, 32, 3))).all()

    # --- step 2 done
    env.step(env_action)
    worker.on_step()
    if window_length == 1:
        assert _equal_data(worker.prev_state, true_rl_states[1])
        assert _equal_data(worker.state, true_rl_states[2])
    else:
        assert _equal_data(
            worker.prev_state,
            true_obs_space_one.encode_stack(
                [
                    true_rl_states[0],
                    true_rl_states[1],
                ]
            ),
        )
        assert _equal_data(
            worker.state,
            true_obs_space_one.encode_stack(
                [
                    true_rl_states[1],
                    true_rl_states[2],
                ]
            ),
        )
    assert _equal_data(worker.state_one_step, true_rl_states[2])

    # render_image
    if render_image_window_length == 1:
        assert (worker.prev_render_image_state == np.ones((64, 32, 3))).all()
        assert (worker.render_image_state == np.ones((64, 32, 3))).all()
    else:
        assert (worker.prev_render_image_state == np.ones((2, 64, 32, 3))).all()
        assert (worker.render_image_state == np.ones((2, 64, 32, 3))).all()
    assert (worker.render_image_state_one_step == np.ones((64, 32, 3))).all()


@pytest.mark.parametrize(
    "env_obs_space, rl_obs_div_num, true_one_space_args, true_space_args, env_states, true_states",
    [
        [_D(5), -1, [1, 0, 4], [2, 0, 4], [1, 2, 3], [[1], [2], [3]]],
        [_AD(2, 0, 5), -1, [2, 0, 5], [4, 0, 5], [[0, 1], [0, 2], [0, 3]], [[0, 1], [0, 2], [0, 3]]],
        [_C(0, 5), -1, [1, 0, 5], [2, 0, 5], [1.2, 2.2, 3.2], [[1], [2], [3]]],
        [_C(0, 5), 6, [1, 0, 6], [2, 0, 6], [1.2, 2.2, 3.2], [[1], [2], [3]]],
        [_AC(2, 0, 5), -1, [2, 0, 5], [4, 0, 5], [[1.1, 2.1], [1.1, 3.1], [1.1, 4.1]], [[1, 2], [1, 3], [1, 4]]],
        [_AC(2, 0, 5), 6, [1, 0, 4], [2, 0, 4], [[1.1, 2.1], [1.1, 3.1], [1.1, 4.1]], [[0], [1], [1]]],
        [
            _B((2, 1), -1, 5),
            -1,
            [2, -1, 5],
            [4, -1, 5],
            [np.array([[1.1], [2.1]]), np.array([[1.1], [3.1]]), np.array([[1.1], [4.1]])],
            [[1, 2], [1, 3], [1, 4]],
        ],
        [
            _B((2, 1), 0, 5, dtype=np.uint8),
            -1,
            [2, 0, 5],
            [4, 0, 5],
            [np.array([[1], [2]]), np.array([[1], [3]]), np.array([[1], [4]])],
            [[1, 2], [1, 3], [1, 4]],
        ],
        [
            _B((2, 1), 0, 5),
            6,
            [1, 0, 4],
            [2, 0, 4],
            [np.array([[1.1], [2.1]]), np.array([[1.1], [3.1]]), np.array([[1.1], [4.1]])],
            [[0], [1], [1]],
        ],
    ],
)
@pytest.mark.parametrize("window_length", [1, 2])
def test_obs_discrete(env_obs_space, rl_obs_div_num, true_one_space_args, true_space_args, env_states, true_states, window_length):
    # list[int], ArrayDiscreteSpace
    _test_obs_episode(
        env_obs_space=env_obs_space,
        rl_obs_type=RLBaseObsTypes.DISCRETE,
        rl_obs_mode="",
        rl_obs_type_override=SpaceTypes.UNKNOWN,
        rl_obs_div_num=rl_obs_div_num,
        window_length=window_length,
        render_image_window_length=2,  # 1とそれ以外はテスト全体でどちらかをcheck
        true_obs_env_space=env_obs_space,
        true_obs_space_one=ArrayDiscreteSpace(*true_one_space_args),
        true_obs_space=ArrayDiscreteSpace(*true_space_args),
        env_states=env_states,
        true_rl_states=true_states,
    )


@pytest.mark.parametrize(
    "env_obs_space, true_one_space_args, true_space_args, env_states, true_states",
    [
        [
            _D(5),
            [(1,), 0, 4, np.float32, SpaceTypes.DISCRETE],
            [(2, 1), 0, 4, np.float32, SpaceTypes.DISCRETE],
            [1, 2, 3],
            [[1], [2], [3]],
        ],
        [
            _AD(3, 0, 5),
            [(3,), 0, 5, np.float32, SpaceTypes.DISCRETE],
            [(2, 3), 0, 5, np.float32, SpaceTypes.DISCRETE],
            [[0, 1, 1], [0, 2, 2], [0, 3, 3]],
            [[0, 1, 1], [0, 2, 2], [0, 3, 3]],
        ],
        [
            _C(0, 5),
            [(1,), 0, 5],
            [(2, 1), 0, 5],
            [1.2, 2.2, 3.2],
            [[1.2], [2.2], [3.2]],
        ],
        [
            _AC(3, 0, 5),
            [(3,), 0, 5],
            [(2, 3), 0, 5],
            [[1.1, 2.1, 2.1], [1.1, 3.1, 3.1], [1.1, 4.1, 4.1]],
            [[1.1, 2.1, 2.1], [1.1, 3.1, 3.1], [1.1, 4.1, 4.1]],
        ],
        [
            _B((3, 1), -1, 5),
            [(3, 1), -1, 5],
            [(2, 3, 1), -1, 5],
            np.array([[[1.1], [2.1], [2.1]], [[1.1], [3.1], [3.1]], [[1.1], [4.1], [4.1]]]),
            [[[1.1], [2.1], [2.1]], [[1.1], [3.1], [3.1]], [[1.1], [4.1], [4.1]]],
        ],
        [
            _B((3, 1), 0, 5, np.uint8),
            [(3, 1), 0, 5, np.float32, SpaceTypes.DISCRETE],
            [(2, 3, 1), 0, 5, np.float32, SpaceTypes.DISCRETE],
            np.array([[[1], [2], [2]], [[1], [3], [3]], [[1], [4], [4]]]),
            [[[1], [2], [2]], [[1], [3], [3]], [[1], [4], [4]]],
        ],
    ],
)
@pytest.mark.parametrize("window_length", [1, 2])
def test_obs_box(env_obs_space, true_one_space_args, true_space_args, env_states, true_states, window_length):
    # NDArray[np.float32] BoxSpace
    _test_obs_episode(
        env_obs_space,
        rl_obs_type=RLBaseObsTypes.BOX,
        rl_obs_mode="",
        rl_obs_type_override=SpaceTypes.UNKNOWN,
        rl_obs_div_num=-1,
        window_length=window_length,
        render_image_window_length=1,
        true_obs_env_space=env_obs_space,
        true_obs_space_one=BoxSpace(*true_one_space_args),
        true_obs_space=BoxSpace(*true_space_args),
        env_states=env_states,
        true_rl_states=[np.array(s, np.float32) for s in true_states],
    )


@pytest.mark.parametrize(
    "env_shape, env_stype",
    [
        [(16, 16, 3), SpaceTypes.COLOR],
        [(16, 16, 5), SpaceTypes.IMAGE],
    ],
)
@pytest.mark.parametrize("window_length", [1, 2])
def test_obs_image(env_shape, env_stype, window_length):
    # NDArray[np.float32] BoxSpace
    _test_obs_episode(
        BoxSpace(env_shape, stype=env_stype),
        rl_obs_type=RLBaseObsTypes.BOX,
        rl_obs_mode="",
        rl_obs_type_override=SpaceTypes.UNKNOWN,
        rl_obs_div_num=-1,
        window_length=window_length,
        render_image_window_length=1,
        true_obs_env_space=BoxSpace(env_shape, stype=env_stype),
        true_obs_space_one=BoxSpace(env_shape, stype=env_stype),
        true_obs_space=BoxSpace((2,) + env_shape, stype=env_stype),
        env_states=[np.ones(env_shape), np.ones(env_shape), np.ones(env_shape)],
        true_rl_states=[np.ones(env_shape), np.ones(env_shape), np.ones(env_shape)],
    )


@pytest.mark.parametrize(
    "env_shape, env_stype",
    [
        [(16, 16), SpaceTypes.GRAY_2ch],
        [(16, 16, 1), SpaceTypes.GRAY_3ch],
    ],
)
@pytest.mark.parametrize("window_length", [1, 2])
def test_obs_image_window_gray_2ch(env_shape, env_stype, window_length):
    _test_obs_episode(
        BoxSpace(env_shape, stype=env_stype),
        rl_obs_type=RLBaseObsTypes.BOX,
        rl_obs_mode="",
        rl_obs_type_override=SpaceTypes.UNKNOWN,
        rl_obs_div_num=-1,
        window_length=window_length,
        render_image_window_length=1,
        true_obs_env_space=BoxSpace(env_shape, stype=env_stype),
        true_obs_space_one=BoxSpace(env_shape, stype=env_stype, is_stack_ch=True),
        true_obs_space=BoxSpace((16, 16, 2), stype=SpaceTypes.IMAGE),
        env_states=[np.ones(env_shape), np.ones(env_shape), np.ones(env_shape)],
        true_rl_states=[np.ones(env_shape), np.ones(env_shape), np.ones(env_shape)],
    )


@pytest.mark.parametrize("window_length", [1, 2])
def test_obs_render_image(window_length):
    _test_obs_episode(
        env_obs_space=DiscreteSpace(1),
        rl_obs_type=RLBaseObsTypes.NONE,
        rl_obs_mode="render_image",
        rl_obs_type_override=SpaceTypes.UNKNOWN,
        rl_obs_div_num=-1,
        window_length=window_length,
        render_image_window_length=1,
        true_obs_env_space=BoxSpace((64, 32, 3), 0, 255, np.uint8, stype=SpaceTypes.COLOR),
        true_obs_space_one=BoxSpace((64, 32, 3), 0, 255, np.uint8, stype=SpaceTypes.COLOR),
        true_obs_space=BoxSpace((2, 64, 32, 3), 0, 255, np.uint8, stype=SpaceTypes.COLOR),
        env_states=[1, 2, 3],
        true_rl_states=[np.ones((64, 32, 3)), np.ones((64, 32, 3)), np.ones((64, 32, 3))],
    )


def test_obs_render_terminal():
    pytest.skip("TODO")


@pytest.mark.parametrize("window_length", [1, 2])
def test_obs_override(window_length):
    _test_obs_episode(
        env_obs_space=BoxSpace((64, 64, 1), stype=SpaceTypes.CONTINUOUS),
        rl_obs_type=RLBaseObsTypes.NONE,
        rl_obs_mode="",
        rl_obs_type_override=SpaceTypes.GRAY_3ch,
        rl_obs_div_num=-1,
        window_length=window_length,
        render_image_window_length=1,
        true_obs_env_space=BoxSpace((64, 64, 1), stype=SpaceTypes.GRAY_3ch),
        true_obs_space_one=BoxSpace((64, 64, 1), stype=SpaceTypes.GRAY_3ch),
        true_obs_space=BoxSpace((64, 64, 2), stype=SpaceTypes.IMAGE),
        env_states=[np.ones((64, 64, 1)), np.ones((64, 64, 1)), np.ones((64, 64, 1))],
        true_rl_states=[np.ones((64, 64, 1)), np.ones((64, 64, 1)), np.ones((64, 64, 1))],
    )
