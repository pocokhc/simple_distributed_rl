from pprint import pprint
from typing import Literal, cast

import numpy as np
import pytest

import srl
from srl.base.context import RunContext
from srl.base.define import RLBaseTypes, SpaceTypes
from srl.base.env.registration import register as register_env
from srl.base.rl.registration import register as register_rl
from srl.base.spaces.array_continuous import ArrayContinuousSpace
from srl.base.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.continuous import ContinuousSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.np_array import NpArraySpace
from srl.base.spaces.space import SpaceBase
from srl.base.spaces.text import TextSpace
from tests.quick.base.rl import worker_run_stub


@pytest.fixture(scope="function", autouse=True)
def scope_function():
    register_env(id="Stub", entry_point=worker_run_stub.__name__ + ":WorkerRunStubEnv", check_duplicate=False)
    register_rl(worker_run_stub.WorkerRunStubRLConfig(), "", "", "", worker_run_stub.__name__ + ":WorkerRunStubRLWorker", check_duplicate=False)
    yield


# ------------------------------------------------------------------
def _equal_data(a, b):
    if isinstance(a, np.ndarray):
        return (a == b).all()
    else:
        return a == b


def _test_obs_episode(
    env_obs_space: SpaceBase,
    rl_obs_type: RLBaseTypes,
    rl_obs_mode: Literal["", "render_image"],
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
    env = srl.make_env(srl.EnvConfig("Stub", {"observation_space": env_obs_space}))
    env_org = cast(worker_run_stub.WorkerRunStubEnv, env.unwrapped)
    env_org.s_states = env_states
    assert len(env_states) == 3

    rl_config = worker_run_stub.WorkerRunStubRLConfig()
    rl_config.enable_assertion = True
    rl_config._observation_type = rl_obs_type
    rl_config.window_length = window_length
    rl_config.observation_mode = rl_obs_mode
    rl_config.observation_division_num = rl_obs_div_num
    rl_config._use_render_image_state = True
    rl_config.render_image_window_length = render_image_window_length

    # --- check rl_config setup
    rl_config.setup(env)
    print(true_obs_space)
    print(rl_config.observation_space)
    print("--")
    print(true_obs_env_space)
    print(rl_config.observation_space_of_env)
    assert rl_config.observation_space_of_env == true_obs_env_space
    print(rl_config.observation_space_one_step)
    print(true_obs_space_one)
    assert rl_config.observation_space_one_step == true_obs_space_one
    if window_length == 1:
        assert rl_config.observation_space == true_obs_space_one
    else:
        assert rl_config.observation_space == true_obs_space

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
    print(worker.state)
    print(true_rl_states[0])
    if window_length == 1:
        assert _equal_data(worker.state, true_rl_states[0])
    else:
        assert _equal_data(
            worker.state,
            true_obs_space_one.encode_stack(
                [
                    true_obs_space_one.get_default(),
                    true_rl_states[0],
                ]
            ),
        )
    assert _equal_data(worker.get_state_one_step(), true_rl_states[0])

    # render_image
    if render_image_window_length == 1:
        assert (worker.render_image_state == np.ones((64, 32, 3))).all()
    else:
        state = np.stack(
            [
                np.zeros((64, 32, 3)),
                np.ones((64, 32, 3)),
            ],
            axis=0,
        )
        assert (worker.render_image_state == state).all()
    assert (worker.get_render_image_state_one_step() == np.ones((64, 32, 3))).all()

    # --- step 1
    env.step(env_action)
    worker.on_step()
    env_action = worker.policy()
    if window_length == 1:
        assert _equal_data(worker.state, true_rl_states[1])
    else:
        assert _equal_data(
            worker.state,
            true_obs_space_one.encode_stack(
                [
                    true_rl_states[0],
                    true_rl_states[1],
                ]
            ),
        )
    assert _equal_data(worker.get_state_one_step(), true_rl_states[1])

    # render_image
    if render_image_window_length == 1:
        assert (worker.render_image_state == np.full((64, 32, 3), 2)).all()
    else:
        state = np.stack(
            [
                np.ones((64, 32, 3)),
                np.full((64, 32, 3), 2),
            ],
            axis=0,
        )
        assert (worker.render_image_state == state).all()
    assert (worker.get_render_image_state_one_step() == np.full((64, 32, 3), 2)).all()

    # --- step 2 done
    env.step(env_action)
    worker.on_step()
    if window_length == 1:
        assert _equal_data(worker.state, true_rl_states[2])
    else:
        assert _equal_data(
            worker.state,
            true_obs_space_one.encode_stack(
                [
                    true_rl_states[1],
                    true_rl_states[2],
                ]
            ),
        )
    assert _equal_data(worker.get_state_one_step(), true_rl_states[2])

    # render_image
    if render_image_window_length == 1:
        assert (worker.render_image_state == np.full((64, 32, 3), 3)).all()
    else:
        state = np.stack(
            [
                np.full((64, 32, 3), 2),
                np.full((64, 32, 3), 3),
            ],
            axis=0,
        )
        assert (worker.render_image_state == state).all()
    assert (worker.get_render_image_state_one_step() == np.full((64, 32, 3), 3)).all()


_params = [
    # --- ARRAY_DISCRETE
    dict(
        env_obs_space=DiscreteSpace(5),
        rl_obs_type=RLBaseTypes.ARRAY_DISCRETE,
        rl_obs_mode="",
        rl_obs_div_num=-1,
        render_image_window_length=2,  # 1とそれ以外はテスト全体でどちらかをcheck
        true_obs_env_space=DiscreteSpace(5),
        true_obs_space_one=ArrayDiscreteSpace(1, 0, 4),
        true_obs_space=ArrayDiscreteSpace(2, 0, 4),
        env_states=[1, 2, 3],
        true_rl_states=[[1], [2], [3]],
    ),
    dict(
        env_obs_space=ArrayDiscreteSpace(2, 0, 5),
        rl_obs_type=RLBaseTypes.ARRAY_DISCRETE,
        rl_obs_mode="",
        rl_obs_div_num=-1,
        render_image_window_length=1,  # 1とそれ以外はテスト全体でどちらかをcheck
        true_obs_env_space=ArrayDiscreteSpace(2, 0, 5),
        true_obs_space_one=ArrayDiscreteSpace(2, 0, 5),
        true_obs_space=ArrayDiscreteSpace(4, 0, 5),
        env_states=[[0, 1], [0, 2], [0, 3]],
        true_rl_states=[[0, 1], [0, 2], [0, 3]],
    ),
    dict(
        env_obs_space=ContinuousSpace(0, 5),
        rl_obs_type=RLBaseTypes.ARRAY_DISCRETE,
        rl_obs_mode="",
        rl_obs_div_num=-1,
        render_image_window_length=2,
        true_obs_env_space=ContinuousSpace(0, 5),
        true_obs_space_one=ArrayDiscreteSpace(1, 0, 5),
        true_obs_space=ArrayDiscreteSpace(2, 0, 5),
        env_states=[1.2, 2.2, 3.2],
        true_rl_states=[[1], [2], [3]],
    ),
    dict(
        env_obs_space=ContinuousSpace(0, 5),
        rl_obs_type=RLBaseTypes.ARRAY_DISCRETE,
        rl_obs_mode="",
        rl_obs_div_num=6,
        render_image_window_length=2,
        true_obs_env_space=ContinuousSpace(0, 5),
        true_obs_space_one=ArrayDiscreteSpace(1, 0, 6),
        true_obs_space=ArrayDiscreteSpace(2, 0, 6),
        env_states=[1.2, 2.2, 3.2],
        true_rl_states=[[1], [2], [3]],
    ),
    dict(
        env_obs_space=ArrayContinuousSpace(2, 0, 5),
        rl_obs_type=RLBaseTypes.ARRAY_DISCRETE,
        rl_obs_mode="",
        rl_obs_div_num=-1,
        render_image_window_length=2,
        true_obs_env_space=ArrayContinuousSpace(2, 0, 5),
        true_obs_space_one=ArrayDiscreteSpace(2, 0, 5),
        true_obs_space=ArrayDiscreteSpace(4, 0, 5),
        env_states=[[1.1, 2.1], [1.1, 3.1], [1.1, 4.1]],
        true_rl_states=[[1, 2], [1, 3], [1, 4]],
    ),
    dict(
        env_obs_space=ArrayContinuousSpace(2, 0, 5),
        rl_obs_type=RLBaseTypes.ARRAY_DISCRETE,
        rl_obs_mode="",
        rl_obs_div_num=6,
        render_image_window_length=2,
        true_obs_env_space=ArrayContinuousSpace(2, 0, 5),
        true_obs_space_one=ArrayDiscreteSpace(1, 0, 4),
        true_obs_space=ArrayDiscreteSpace(2, 0, 4),
        env_states=[[1.1, 2.1], [1.1, 3.1], [1.1, 4.1]],
        true_rl_states=[[0], [1], [1]],
    ),
    dict(
        env_obs_space=NpArraySpace(2, 0, 5),
        rl_obs_type=RLBaseTypes.ARRAY_DISCRETE,
        rl_obs_mode="",
        rl_obs_div_num=-1,
        render_image_window_length=2,
        true_obs_env_space=NpArraySpace(2, 0, 5),
        true_obs_space_one=ArrayDiscreteSpace(2, 0, 5),
        true_obs_space=ArrayDiscreteSpace(4, 0, 5),
        env_states=[[1.1, 2.1], [1.1, 3.1], [1.1, 4.1]],
        true_rl_states=[[1, 2], [1, 3], [1, 4]],
    ),
    dict(
        env_obs_space=NpArraySpace(2, 0, 5),
        rl_obs_type=RLBaseTypes.ARRAY_DISCRETE,
        rl_obs_mode="",
        rl_obs_div_num=6,
        render_image_window_length=2,
        true_obs_env_space=NpArraySpace(2, 0, 5),
        true_obs_space_one=ArrayDiscreteSpace(1, 0, 4),
        true_obs_space=ArrayDiscreteSpace(2, 0, 4),
        env_states=[[1.1, 2.1], [1.1, 3.1], [1.1, 4.1]],
        true_rl_states=[[0], [1], [1]],
    ),
    dict(
        env_obs_space=BoxSpace((2, 1), -1, 5),
        rl_obs_type=RLBaseTypes.ARRAY_DISCRETE,
        rl_obs_mode="",
        rl_obs_div_num=-1,
        render_image_window_length=2,
        true_obs_env_space=BoxSpace((2, 1), -1, 5),
        true_obs_space_one=ArrayDiscreteSpace(2, -1, 5),
        true_obs_space=ArrayDiscreteSpace(4, -1, 5),
        env_states=[np.array([[1.1], [2.1]]), np.array([[1.1], [3.1]]), np.array([[1.1], [4.1]])],
        true_rl_states=[[1, 2], [1, 3], [1, 4]],
    ),
    dict(
        env_obs_space=BoxSpace((2, 1), 0, 5),
        rl_obs_type=RLBaseTypes.ARRAY_DISCRETE,
        rl_obs_mode="",
        rl_obs_div_num=6,
        render_image_window_length=2,
        true_obs_env_space=BoxSpace((2, 1), 0, 5),
        true_obs_space_one=ArrayDiscreteSpace(1, 0, 4),
        true_obs_space=ArrayDiscreteSpace(2, 0, 4),
        env_states=[np.array([[1.1], [2.1]]), np.array([[1.1], [3.1]]), np.array([[1.1], [4.1]])],
        true_rl_states=[[0], [1], [1]],
    ),
    dict(
        env_obs_space=TextSpace(max_length=2),
        rl_obs_type=RLBaseTypes.ARRAY_DISCRETE,
        rl_obs_mode="",
        rl_obs_div_num=-1,
        render_image_window_length=2,
        true_obs_env_space=TextSpace(max_length=2),
        true_obs_space_one=ArrayDiscreteSpace(2, 0, 127),
        true_obs_space=ArrayDiscreteSpace(4, 0, 127),
        env_states=["12", "34", "56"],
        true_rl_states=[[49, 50], [51, 52], [53, 54]],
    ),
    # --- BOX
    dict(
        env_obs_space=DiscreteSpace(5),
        rl_obs_type=RLBaseTypes.BOX,
        rl_obs_mode="",
        rl_obs_div_num=-1,
        render_image_window_length=2,
        true_obs_env_space=DiscreteSpace(5),
        true_obs_space_one=BoxSpace((1,), 0, 4, np.float32, SpaceTypes.DISCRETE),
        true_obs_space=BoxSpace((2, 1), 0, 4, np.float32, SpaceTypes.DISCRETE),
        env_states=[1, 2, 3],
        true_rl_states=[np.array([1], np.float32), np.array([2], np.float32), np.array([3], np.float32)],
    ),
    dict(
        env_obs_space=ArrayDiscreteSpace(3, 0, 5),
        rl_obs_type=RLBaseTypes.BOX,
        rl_obs_mode="",
        rl_obs_div_num=-1,
        render_image_window_length=2,
        true_obs_env_space=ArrayDiscreteSpace(3, 0, 5),
        true_obs_space_one=BoxSpace((3,), 0, 5, np.float32, SpaceTypes.DISCRETE),
        true_obs_space=BoxSpace((2, 3), 0, 5, np.float32, SpaceTypes.DISCRETE),
        env_states=[[0, 1, 1], [0, 2, 2], [0, 3, 3]],
        true_rl_states=[np.array([0, 1, 1], np.float32), np.array([0, 2, 2], np.float32), np.array([0, 3, 3], np.float32)],
    ),
    dict(
        env_obs_space=ContinuousSpace(0, 5),
        rl_obs_type=RLBaseTypes.BOX,
        rl_obs_mode="",
        rl_obs_div_num=-1,
        render_image_window_length=2,
        true_obs_env_space=ContinuousSpace(0, 5),
        true_obs_space_one=BoxSpace((1,), 0, 5),
        true_obs_space=BoxSpace((2, 1), 0, 5),
        env_states=[1.2, 2.2, 3.2],
        true_rl_states=[np.array([1.2], np.float32), np.array([2.2], np.float32), np.array([3.2], np.float32)],
    ),
    dict(
        env_obs_space=ArrayContinuousSpace(3, 0, 5),
        rl_obs_type=RLBaseTypes.BOX,
        rl_obs_mode="",
        rl_obs_div_num=-1,
        render_image_window_length=2,
        true_obs_env_space=ArrayContinuousSpace(3, 0, 5),
        true_obs_space_one=BoxSpace((3,), 0, 5),
        true_obs_space=BoxSpace((2, 3), 0, 5),
        env_states=[[1.1, 2.1, 2.1], [1.1, 3.1, 3.1], [1.1, 4.1, 4.1]],
        true_rl_states=[np.array([1.1, 2.1, 2.1], np.float32), np.array([1.1, 3.1, 3.1], np.float32), np.array([1.1, 4.1, 4.1], np.float32)],
    ),
    dict(
        env_obs_space=NpArraySpace(3, 0, 5),
        rl_obs_type=RLBaseTypes.BOX,
        rl_obs_mode="",
        rl_obs_div_num=-1,
        render_image_window_length=2,
        true_obs_env_space=NpArraySpace(3, 0, 5),
        true_obs_space_one=BoxSpace((3,), 0, 5),
        true_obs_space=BoxSpace((2, 3), 0, 5),
        env_states=[np.array([1.1, 2.1, 2.1], np.float32), np.array([1.1, 3.1, 3.1], np.float32), np.array([1.1, 4.1, 4.1], np.float32)],
        true_rl_states=[np.array([1.1, 2.1, 2.1], np.float32), np.array([1.1, 3.1, 3.1], np.float32), np.array([1.1, 4.1, 4.1], np.float32)],
    ),
    dict(  # a
        env_obs_space=BoxSpace((3, 1), -1, 5),
        rl_obs_type=RLBaseTypes.BOX,
        rl_obs_mode="",
        rl_obs_div_num=-1,
        render_image_window_length=2,
        true_obs_env_space=BoxSpace((3, 1), -1, 5),
        true_obs_space_one=BoxSpace((3, 1), -1, 5),
        true_obs_space=BoxSpace((2, 3, 1), -1, 5),
        env_states=np.array([[[1.1], [2.1], [2.1]], [[1.1], [3.1], [3.1]], [[1.1], [4.1], [4.1]]], np.float32),
        true_rl_states=np.array([[[1.1], [2.1], [2.1]], [[1.1], [3.1], [3.1]], [[1.1], [4.1], [4.1]]], np.float32),
    ),
    dict(
        env_obs_space=BoxSpace((3, 1), 0, 5, np.uint8),
        rl_obs_type=RLBaseTypes.BOX,
        rl_obs_mode="",
        rl_obs_div_num=-1,
        render_image_window_length=2,
        true_obs_env_space=BoxSpace((3, 1), 0, 5, np.uint8),
        true_obs_space_one=BoxSpace((3, 1), 0, 5, np.float32, SpaceTypes.DISCRETE),
        true_obs_space=BoxSpace((2, 3, 1), 0, 5, np.float32, SpaceTypes.DISCRETE),
        env_states=np.array([[[1], [2], [2]], [[1], [3], [3]], [[1], [4], [4]]], np.float32),
        true_rl_states=np.array([[[1], [2], [2]], [[1], [3], [3]], [[1], [4], [4]]], np.float32),
    ),
    # IMAGE
    dict(
        env_obs_space=BoxSpace((16, 16), stype=SpaceTypes.GRAY_2ch),
        rl_obs_type=RLBaseTypes.BOX,
        rl_obs_mode="",
        rl_obs_div_num=-1,
        render_image_window_length=2,
        true_obs_env_space=BoxSpace((16, 16), stype=SpaceTypes.GRAY_2ch),
        true_obs_space_one=BoxSpace((16, 16), stype=SpaceTypes.GRAY_2ch, is_stack_ch=True),
        true_obs_space=BoxSpace((16, 16, 2), stype=SpaceTypes.IMAGE),
        env_states=[np.ones((16, 16)), np.ones((16, 16)), np.ones((16, 16))],
        true_rl_states=[np.ones((16, 16)), np.ones((16, 16)), np.ones((16, 16))],
    ),
    dict(
        env_obs_space=BoxSpace((16, 16, 1), stype=SpaceTypes.GRAY_3ch),
        rl_obs_type=RLBaseTypes.BOX,
        rl_obs_mode="",
        rl_obs_div_num=-1,
        render_image_window_length=2,
        true_obs_env_space=BoxSpace((16, 16, 1), stype=SpaceTypes.GRAY_3ch),
        true_obs_space_one=BoxSpace((16, 16, 1), stype=SpaceTypes.GRAY_3ch, is_stack_ch=True),
        true_obs_space=BoxSpace((16, 16, 2), stype=SpaceTypes.IMAGE),
        env_states=[np.ones((16, 16, 1)), np.ones((16, 16, 1)), np.ones((16, 16, 1))],
        true_rl_states=[np.ones((16, 16, 1)), np.ones((16, 16, 1)), np.ones((16, 16, 1))],
    ),
    dict(
        env_obs_space=BoxSpace((16, 16, 3), stype=SpaceTypes.COLOR),
        rl_obs_type=RLBaseTypes.BOX,
        rl_obs_mode="",
        rl_obs_div_num=-1,
        render_image_window_length=2,
        true_obs_env_space=BoxSpace((16, 16, 3), stype=SpaceTypes.COLOR),
        true_obs_space_one=BoxSpace((16, 16, 3), stype=SpaceTypes.COLOR),
        true_obs_space=BoxSpace((2, 16, 16, 3), stype=SpaceTypes.COLOR),
        env_states=[np.ones((16, 16, 3)), np.ones((16, 16, 3)), np.ones((16, 16, 3))],
        true_rl_states=[np.ones((16, 16, 3)), np.ones((16, 16, 3)), np.ones((16, 16, 3))],
    ),
    dict(
        env_obs_space=BoxSpace((16, 16, 5), stype=SpaceTypes.IMAGE),
        rl_obs_type=RLBaseTypes.BOX,
        rl_obs_mode="",
        rl_obs_div_num=-1,
        render_image_window_length=2,
        true_obs_env_space=BoxSpace((16, 16, 5), stype=SpaceTypes.IMAGE),
        true_obs_space_one=BoxSpace((16, 16, 5), stype=SpaceTypes.IMAGE),
        true_obs_space=BoxSpace((2, 16, 16, 5), stype=SpaceTypes.IMAGE),
        env_states=[np.ones((16, 16, 5)), np.ones((16, 16, 5)), np.ones((16, 16, 5))],
        true_rl_states=[np.ones((16, 16, 5)), np.ones((16, 16, 5)), np.ones((16, 16, 5))],
    ),
    dict(
        env_obs_space=TextSpace(max_length=1),
        rl_obs_type=RLBaseTypes.BOX,
        rl_obs_mode="",
        rl_obs_div_num=-1,
        render_image_window_length=2,
        true_obs_env_space=TextSpace(max_length=1),
        true_obs_space_one=BoxSpace((1,), 0, 127, np.float32, SpaceTypes.DISCRETE),
        true_obs_space=BoxSpace((2, 1), 0, 127, np.float32, SpaceTypes.DISCRETE),
        env_states=["1", "2", "3"],
        true_rl_states=[np.array([49]), np.array([50]), np.array([51])],
    ),
    # --- render_image
    dict(
        env_obs_space=DiscreteSpace(1),
        rl_obs_type=RLBaseTypes.NONE,
        rl_obs_mode="render_image",
        rl_obs_div_num=-1,
        render_image_window_length=1,
        true_obs_env_space=BoxSpace((64, 32, 3), 0, 255, np.uint8, stype=SpaceTypes.COLOR),
        true_obs_space_one=BoxSpace((64, 32, 3), 0, 255, np.uint8, stype=SpaceTypes.COLOR),
        true_obs_space=BoxSpace((2, 64, 32, 3), 0, 255, np.uint8, stype=SpaceTypes.COLOR),
        env_states=[1, 2, 3],
        true_rl_states=[
            np.full((64, 32, 3), 1),
            np.full((64, 32, 3), 2),
            np.full((64, 32, 3), 3),
        ],
    ),
]


@pytest.mark.parametrize("kwargs", _params)
@pytest.mark.parametrize("window_length", [1, 2])
def test_obs(kwargs, window_length):
    kwargs["window_length"] = window_length
    pprint(kwargs)
    _test_obs_episode(**kwargs)


def test_obs_render_terminal():
    pytest.skip("TODO")
