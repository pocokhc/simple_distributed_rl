from pprint import pprint
from typing import cast

import numpy as np
import pytest

import srl
from srl.base.context import RunContext
from srl.base.define import RLBaseTypes, SpaceTypes
from srl.base.env.registration import register as register_env
from srl.base.rl.registration import register as register_rl
from srl.base.spaces.array_continuous import ArrayContinuousSpace
from srl.base.spaces.array_continuous_list import ArrayContinuousListSpace
from srl.base.spaces.array_discrete import ArrayDiscreteSpace
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.continuous import ContinuousSpace
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.space import SpaceBase
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


def _test_action_episode(
    env_act_space: SpaceBase,
    rl_act_type: RLBaseTypes,
    rl_act_type_override: RLBaseTypes,
    true_act_space: SpaceBase,
    rl_act,
    true_env_act,
):
    env = srl.make_env(srl.EnvConfig("Stub", {"action_space": env_act_space, "invalid_action": False}))
    env_org = cast(worker_run_stub.WorkerRunStubEnv, env.unwrapped)
    env_org.s_states = [1, 2, 3]  # 2step

    rl_config = worker_run_stub.WorkerRunStubRLConfig()
    rl_config.enable_assertion = True
    rl_config._action_type = rl_act_type
    rl_config.override_action_type = rl_act_type_override

    # --- check rl_config setup
    rl_config.setup(env)
    print(rl_config.action_space)
    print(true_act_space)
    assert rl_config.action_space == true_act_space

    # --- setup
    worker = srl.make_worker(rl_config, env)
    worker_base = cast(worker_run_stub.WorkerRunStubRLWorker, worker.worker)
    worker_base.action = rl_act

    context = RunContext()
    env.setup(context)
    worker.setup(context)

    # --- reset
    env.reset()
    worker.reset(0)
    assert _equal_data(worker.action, true_act_space.get_default())
    assert worker.invalid_actions == []
    assert worker.reward == 0.0
    assert not worker.done
    assert not worker.terminated

    env_action = worker.policy()
    assert _equal_data(worker.action, rl_act)
    assert env_act_space.check_val(env_action)
    assert _equal_data(env_action, true_env_act)
    # assert worker.invalid_actions == [[1.0]]
    assert worker.reward == 0.0
    assert not worker.done
    assert not worker.terminated

    # step1
    env.step(env_action)
    worker.on_step()
    env_action = worker.policy()
    assert _equal_data(worker.action, rl_act)
    assert env_act_space.check_val(env_action)
    assert _equal_data(env_action, true_env_act)
    # assert worker.invalid_actions == [[2.0]]
    assert worker.reward == 1.0
    assert not worker.done
    assert not worker.terminated

    # step2 done
    env.step(env_action)
    worker.on_step()
    assert _equal_data(worker.action, rl_act)
    assert env_act_space.check_val(env_action)
    assert _equal_data(env_action, true_env_act)
    # assert worker.invalid_actions == [[3.0]]
    assert worker.reward == 2.0
    assert worker.done
    assert worker.terminated


_params = [
    # --- DISCRETE
    dict(
        env_act_space=DiscreteSpace(5),
        rl_act_type=RLBaseTypes.DISCRETE,
        rl_act_type_override=RLBaseTypes.NONE,
        true_act_space=DiscreteSpace(5),
        rl_act=2,
        true_env_act=2,
    ),
    dict(
        env_act_space=ArrayDiscreteSpace(2, -1, 1),
        rl_act_type=RLBaseTypes.DISCRETE,
        rl_act_type_override=RLBaseTypes.NONE,
        true_act_space=DiscreteSpace(3 * 3),
        rl_act=2,
        true_env_act=[-1, 1],
    ),
    dict(
        env_act_space=ContinuousSpace(0, 1),
        rl_act_type=RLBaseTypes.DISCRETE,
        rl_act_type_override=RLBaseTypes.NONE,
        true_act_space=DiscreteSpace(10),
        rl_act=2,
        true_env_act=0.2222222222222222,
    ),
    dict(
        env_act_space=ArrayContinuousListSpace(2, -1, 1),
        rl_act_type=RLBaseTypes.DISCRETE,
        rl_act_type_override=RLBaseTypes.NONE,
        true_act_space=DiscreteSpace(9),
        rl_act=0,
        true_env_act=[-1.0, -1.0],
    ),
    dict(
        env_act_space=ArrayContinuousSpace(2, -1, 1),
        rl_act_type=RLBaseTypes.DISCRETE,
        rl_act_type_override=RLBaseTypes.NONE,
        true_act_space=DiscreteSpace(9),
        rl_act=0,
        true_env_act=[-1.0, -1.0],
    ),
    dict(
        env_act_space=BoxSpace((2, 1), -1, 1),
        rl_act_type=RLBaseTypes.DISCRETE,
        rl_act_type_override=RLBaseTypes.NONE,
        true_act_space=DiscreteSpace(9),
        rl_act=0,
        true_env_act=np.array([[-1], [-1]], np.float32),
    ),
    # --- ArrayContinuousSpace
    dict(
        env_act_space=DiscreteSpace(5),
        rl_act_type=RLBaseTypes.ARRAY_CONTINUOUS_LIST,
        rl_act_type_override=RLBaseTypes.NONE,
        true_act_space=ArrayContinuousListSpace(1, 0, 4),
        rl_act=[2.0],
        true_env_act=2,
    ),
    dict(
        env_act_space=ArrayDiscreteSpace(2, -1, 1),
        rl_act_type=RLBaseTypes.ARRAY_CONTINUOUS_LIST,
        rl_act_type_override=RLBaseTypes.NONE,
        true_act_space=ArrayContinuousListSpace(2, -1, 1),
        rl_act=[1.0, 1.0],
        true_env_act=[1, 1],
    ),
    dict(
        env_act_space=ContinuousSpace(0, 1),
        rl_act_type=RLBaseTypes.ARRAY_CONTINUOUS_LIST,
        rl_act_type_override=RLBaseTypes.NONE,
        true_act_space=ArrayContinuousListSpace(1, 0, 1),
        rl_act=[1.0],
        true_env_act=1.0,
    ),
    dict(
        env_act_space=ArrayContinuousListSpace(1, 0, 1),
        rl_act_type=RLBaseTypes.ARRAY_CONTINUOUS_LIST,
        rl_act_type_override=RLBaseTypes.NONE,
        true_act_space=ArrayContinuousListSpace(1, 0, 1),
        rl_act=[1.0],
        true_env_act=[1.0],
    ),
    dict(
        env_act_space=ArrayContinuousSpace(1, 0, 1),
        rl_act_type=RLBaseTypes.ARRAY_CONTINUOUS_LIST,
        rl_act_type_override=RLBaseTypes.NONE,
        true_act_space=ArrayContinuousListSpace(1, 0, 1),
        rl_act=[1.0],
        true_env_act=np.array([1.0], dtype=np.float32),
    ),
    dict(
        env_act_space=BoxSpace((2, 1), -1, 1),
        rl_act_type=RLBaseTypes.ARRAY_CONTINUOUS_LIST,
        rl_act_type_override=RLBaseTypes.NONE,
        true_act_space=ArrayContinuousListSpace(2, -1, 1),
        rl_act=[1.0, 1.0],
        true_env_act=np.array([[1.0], [1.0]], np.float32),
    ),
    # --- IMAGE
    dict(
        env_act_space=BoxSpace((64, 64), 0, 1, stype=SpaceTypes.GRAY_2ch),
        rl_act_type=RLBaseTypes.NONE,
        rl_act_type_override=RLBaseTypes.NONE,
        true_act_space=BoxSpace((64, 64), 0, 1, stype=SpaceTypes.GRAY_2ch),
        rl_act=np.zeros((64, 64), np.float32),
        true_env_act=np.zeros((64, 64), np.float32),
    ),
    dict(
        env_act_space=BoxSpace((64, 64), 0, 1, stype=SpaceTypes.GRAY_2ch),
        rl_act_type=RLBaseTypes.GRAY_3ch,
        rl_act_type_override=RLBaseTypes.NONE,
        true_act_space=BoxSpace((64, 64, 1), 0, 1, stype=SpaceTypes.GRAY_3ch),
        rl_act=np.zeros((64, 64, 1), np.float32),
        true_env_act=np.zeros((64, 64), np.float32),
    ),
    # --- DISC | COUNT
    dict(
        env_act_space=DiscreteSpace(5),
        rl_act_type=RLBaseTypes.DISCRETE | RLBaseTypes.ARRAY_CONTINUOUS,
        rl_act_type_override=RLBaseTypes.NONE,
        true_act_space=DiscreteSpace(5),
        rl_act=2,
        true_env_act=2,
    ),
    dict(
        env_act_space=ArrayDiscreteSpace(2, -1, 1),
        rl_act_type=RLBaseTypes.DISCRETE | RLBaseTypes.ARRAY_CONTINUOUS,
        rl_act_type_override=RLBaseTypes.NONE,
        true_act_space=DiscreteSpace(3 * 3),
        rl_act=2,
        true_env_act=[-1, 1],
    ),
    dict(
        env_act_space=ContinuousSpace(0, 1),
        rl_act_type=RLBaseTypes.DISCRETE | RLBaseTypes.ARRAY_CONTINUOUS,
        rl_act_type_override=RLBaseTypes.NONE,
        true_act_space=ArrayContinuousSpace(1, 0, 1),
        rl_act=np.array([1.0], np.float32),
        true_env_act=1.0,
    ),
    dict(
        env_act_space=ArrayContinuousListSpace(2, -1, 1),
        rl_act_type=RLBaseTypes.DISCRETE | RLBaseTypes.ARRAY_CONTINUOUS,
        rl_act_type_override=RLBaseTypes.NONE,
        true_act_space=ArrayContinuousSpace(2, -1, 1),
        rl_act=np.array([-1.0, -0.5], np.float32),
        true_env_act=[-1.0, -0.5],
    ),
    dict(
        env_act_space=ArrayContinuousSpace(2, -1, 1),
        rl_act_type=RLBaseTypes.DISCRETE | RLBaseTypes.ARRAY_CONTINUOUS,
        rl_act_type_override=RLBaseTypes.NONE,
        true_act_space=ArrayContinuousSpace(2, -1, 1),
        rl_act=np.array([-1.0, -0.5], np.float32),
        true_env_act=np.array([-1.0, -0.5], np.float32),
    ),
    dict(
        env_act_space=BoxSpace((2, 1), -1, 1),
        rl_act_type=RLBaseTypes.DISCRETE | RLBaseTypes.ARRAY_CONTINUOUS,
        rl_act_type_override=RLBaseTypes.NONE,
        true_act_space=ArrayContinuousSpace(2, -1, 1),
        rl_act=np.array([-1.0, -0.5], np.float32),
        true_env_act=np.array([[-1], [-0.5]], np.float32),
    ),
]


@pytest.mark.parametrize("kwargs", _params)
def test_action(kwargs):
    pprint(kwargs)
    _test_action_episode(**kwargs)


# --------------------------------------


@pytest.mark.parametrize(
    "rl_act_type",
    [
        RLBaseTypes.NONE,
        RLBaseTypes.DISCRETE,
        RLBaseTypes.ARRAY_CONTINUOUS,
        RLBaseTypes.DISCRETE | RLBaseTypes.ARRAY_CONTINUOUS,
    ],
)
@pytest.mark.parametrize(
    "env_act_space",
    [
        DiscreteSpace(5),
        ArrayDiscreteSpace(2, -1, 1),
        ContinuousSpace(0, 1),
        ArrayContinuousListSpace(2, -1, 1),
        ArrayContinuousSpace(2, -1, 1),
        BoxSpace((2, 1), -1, 1),
        BoxSpace((2, 1), -1, 1, np.int8),
        BoxSpace((2, 1), -1, 1, stype=SpaceTypes.COLOR),
    ],
)
def test_sample_action(env_act_space, rl_act_type):
    if env_act_space.is_image() and (rl_act_type & RLBaseTypes.DISCRETE):
        pytest.skip("intに変換できない")
    env = srl.make_env(srl.EnvConfig("Stub", {"action_space": env_act_space}))

    rl_config = worker_run_stub.WorkerRunStubRLConfig()
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
        [ArrayContinuousListSpace(1), True],
        [ArrayContinuousListSpace(1, 0, 5), False],
        [ArrayContinuousSpace(1), True],
        [ArrayContinuousSpace(1, 0, 5), False],
        [BoxSpace((1,)), True],
        [BoxSpace((1,), -1, 1), False],
    ],
)
def test_sample_action_to_env(env_act_space, is_raise):
    env = srl.make_env(srl.EnvConfig("Stub", {"action_space": env_act_space}))
    rl_config = worker_run_stub.WorkerRunStubRLConfig()

    if is_raise:  # rangeの定義がない場合actionを定義できない
        with pytest.raises(AssertionError):
            worker_run = srl.make_worker(rl_config, env)
        return
    worker_run = srl.make_worker(rl_config, env)

    print(worker_run.config.action_space)
    for _ in range(100):
        action = worker_run.sample_action_to_env()
        print(action)
        assert env_act_space.check_val(action)
