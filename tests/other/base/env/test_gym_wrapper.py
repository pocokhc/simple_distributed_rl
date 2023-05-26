import math
import random

import numpy as np
import pytest

import srl
from srl.base.define import EnvObservationTypes
from srl.base.spaces import ArrayDiscreteSpace, BoxSpace
from srl.test import TestEnv


def test_play_FrozenLake():
    pytest.importorskip("gym")

    # observation_space: Discrete(16)
    # action_space     : Discrete(4)
    tester = TestEnv()
    env = tester.play_test("FrozenLake-v1")
    assert env.observation_type == EnvObservationTypes.DISCRETE
    assert isinstance(env.observation_space, ArrayDiscreteSpace)
    assert isinstance(env.action_space, ArrayDiscreteSpace)
    env.observation_space.assert_params(1, [0], [15])
    env.action_space.assert_params(1, [0], [3])


def test_play_CartPole():
    pytest.importorskip("gym")

    # observation_space: Box((4,))
    # action_space     : Discrete(2)
    tester = TestEnv()
    env = tester.play_test("CartPole-v1", max_step=10)
    assert env.observation_type == EnvObservationTypes.CONTINUOUS
    assert isinstance(env.observation_space, BoxSpace)
    assert isinstance(env.action_space, ArrayDiscreteSpace)
    assert env.observation_space.shape == (4,)
    env.action_space.assert_params(1, [0], [1])


def test_play_Blackjack():
    pytest.importorskip("gym")

    # observation_space: Tuple(Discrete(32), Discrete(11), Discrete(2))
    # action_space     : Discrete(2)
    tester = TestEnv()
    env = tester.play_test("Blackjack-v1", max_step=10)
    assert env.observation_type == EnvObservationTypes.DISCRETE
    assert isinstance(env.observation_space, ArrayDiscreteSpace)
    assert isinstance(env.action_space, ArrayDiscreteSpace)
    env.observation_space.assert_params(3, [0, 0, 0], [31, 10, 1])
    env.action_space.assert_params(1, [0], [1])


def test_play_Pendulum():
    pytest.importorskip("gym")

    # observation_space: Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)
    # action_space     : Box(-2.0, 2.0, (1,), float32)
    tester = TestEnv()
    env = tester.play_test("Pendulum-v1", max_step=10)
    assert env.observation_type == EnvObservationTypes.CONTINUOUS
    assert isinstance(env.observation_space, BoxSpace)
    assert isinstance(env.action_space, BoxSpace)
    env.observation_space.assert_params((3,), np.array([-1, -1, -8]), np.array([1, 1, 8]))
    env.action_space.assert_params((1,), np.array([-2]), np.array([2]))


def test_play_Tetris():
    pytest.importorskip("gym")
    pytest.importorskip("ale_py")

    # Box(0, 255, (210, 160, 3), uint8)
    # Discrete(5)
    tester = TestEnv()
    env = tester.play_test("ALE/Tetris-v5", check_render=False, max_step=10)
    assert env.observation_type == EnvObservationTypes.COLOR
    assert isinstance(env.observation_space, BoxSpace)
    assert isinstance(env.action_space, ArrayDiscreteSpace)
    env.observation_space.assert_params((210, 160, 3), np.zeros((210, 160, 3)), np.full((210, 160, 3), 255))
    env.action_space.assert_params(1, [0], [4])


def test_play_Tetris_ram():
    pytest.importorskip("gym")
    pytest.importorskip("ale_py")

    # Box(0, 255, (128,), uint8)
    # Discrete(5)
    tester = TestEnv()
    env = tester.play_test("ALE/Tetris-ram-v5", check_render=False, max_step=10)
    assert env.observation_type == EnvObservationTypes.DISCRETE
    assert isinstance(env.observation_space, BoxSpace)
    assert isinstance(env.action_space, ArrayDiscreteSpace)
    env.observation_space.assert_params((128,), np.array((0,) * 128), np.array((255,) * 128))
    env.action_space.assert_params(1, [0], [4])


# 時間がかかる割に有益じゃないのでコメントアウト
# def test_play_all(self):
#     import gym
#     import gym.error
#     from gym import envs
#     from tqdm import tqdm
#
#     specs = envs.registry.all()
#
#     for spec in tqdm(list(reversed(list(specs)))):
#         try:
#             gym.make(spec.id)
#             self.tester.play_test(spec.id, check_render=False, max_step=5)
#         except AttributeError:
#             pass
#         except gym.error.DependencyNotInstalled:
#             pass  # No module named 'mujoco_py'
#         except ModuleNotFoundError:
#             pass  # unsupported env
#         except Exception:
#             print(spec.id)
#             raise

# --------------------------------


def test_space():
    pytest.importorskip("gym")

    from gym import spaces

    from srl.base.env.gym_wrapper import gym_space_flatten, gym_space_flatten_decode, gym_space_flatten_encode

    space = spaces.Dict(
        {
            "ext_controller": spaces.MultiDiscrete([5, 2, 2]),
            "inner_state": spaces.Dict(
                {
                    "charge": spaces.Discrete(100),
                    "system_checks": spaces.MultiBinary(10),
                    "job_status": spaces.Dict(
                        {
                            "task": spaces.Discrete(5),
                            "progress": spaces.Box(low=0, high=100, shape=()),
                        }
                    ),
                }
            ),
            "other": spaces.Tuple(
                [
                    spaces.Discrete(7),
                    spaces.Box(low=0, high=10, shape=(2, 3)),
                ]
            ),
        }
    )

    flat_space, is_discrete = gym_space_flatten(space)
    print(flat_space)
    assert not is_discrete
    assert isinstance(flat_space, BoxSpace)
    assert flat_space.shape == (23,)
    assert (flat_space.low == [0] * 23).all()
    assert (
        flat_space.high
        == [
            5.0,
            2.0,
            2.0,
            99.0,
            100.0,
            4.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            6.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
        ]
    ).all()

    val = space.sample()
    print(val)

    encode_val = gym_space_flatten_encode(space, val)
    print(encode_val)
    assert len(encode_val) == 23

    decode_val = gym_space_flatten_decode(space, encode_val)
    print(decode_val)

    print(val["ext_controller"], decode_val["ext_controller"])
    assert (val["ext_controller"] == decode_val["ext_controller"]).all()
    assert val["inner_state"]["charge"] == decode_val["inner_state"]["charge"]
    assert val["inner_state"]["job_status"]["progress"] == decode_val["inner_state"]["job_status"]["progress"]
    assert val["inner_state"]["job_status"]["task"] == decode_val["inner_state"]["job_status"]["task"]
    assert (val["inner_state"]["system_checks"] == decode_val["inner_state"]["system_checks"]).all()
    assert val["other"][0] == decode_val["other"][0]
    assert (val["other"][1] == decode_val["other"][1]).all()


def test_space_discrete():
    pytest.importorskip("gym")

    from gym import spaces

    from srl.base.env.gym_wrapper import gym_space_flatten, gym_space_flatten_decode, gym_space_flatten_encode

    space = spaces.Dict(
        {
            "ext_controller": spaces.MultiDiscrete([5, 2, 2]),
            "inner_state": spaces.Dict(
                {
                    "charge": spaces.Discrete(100),
                    "system_checks": spaces.MultiBinary([3, 2]),
                    "job_status": spaces.Dict(
                        {
                            "task": spaces.Discrete(5),
                        }
                    ),
                }
            ),
            "other": spaces.Tuple(
                [
                    spaces.Discrete(7),
                ]
            ),
        }
    )

    flat_space, is_discrete = gym_space_flatten(space)
    print(flat_space)
    print(flat_space.high)
    assert is_discrete
    assert isinstance(flat_space, ArrayDiscreteSpace)
    assert flat_space.list_size == 12
    assert flat_space.list_low == [0] * 12
    assert flat_space.list_high == [
        5,
        2,
        2,
        99,
        4,
        1,
        1,
        1,
        1,
        1,
        1,
        6,
    ]

    val = space.sample()
    print(val)

    encode_val = gym_space_flatten_encode(space, val)
    print(encode_val)
    assert len(encode_val) == 12

    decode_val = gym_space_flatten_decode(space, encode_val)
    print(decode_val)

    print(val["ext_controller"], decode_val["ext_controller"])
    assert (val["ext_controller"] == decode_val["ext_controller"]).all()
    assert val["inner_state"]["charge"] == decode_val["inner_state"]["charge"]
    assert val["inner_state"]["job_status"]["task"] == decode_val["inner_state"]["job_status"]["task"]
    assert (val["inner_state"]["system_checks"] == decode_val["inner_state"]["system_checks"]).all()
    assert val["other"][0] == decode_val["other"][0]


def test_random():
    pytest.importorskip("gym")

    env = srl.make_env("Pendulum-v1")

    seed = 1
    true_reward = -2.4091601371765137

    random.seed(seed)
    np.random.seed(seed)

    reward = 0
    env.reset(seed=seed)
    for _ in range(10):
        env.step(env.sample())
        reward += env.reward
    assert math.isclose(reward, true_reward)

    random.seed(seed)
    np.random.seed(seed)

    reward = 0
    env.reset(seed=seed)
    for _ in range(10):
        env.step(env.sample())
        reward += env.reward
    assert math.isclose(reward, true_reward)
