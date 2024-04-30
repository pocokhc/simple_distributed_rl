from typing import cast

import numpy as np

import srl
from srl.base.define import SpaceTypes
from srl.base.rl.config import DummyRLConfig
from srl.base.spaces.box import BoxSpace  # noqa F401
from srl.envs import grid
from srl.test.env import TestEnv


def test_grid():
    tester = TestEnv()
    tester.play_test("Grid")


def test_easy_grid():
    tester = TestEnv()
    tester.play_test("EasyGrid")


def test_processor():
    processor = grid.LayerProcessor()
    env = srl.make_env("Grid")
    env.setup()
    env.reset()

    env_org: grid.Grid = env.unwrapped
    field = np.zeros((env_org.H, env_org.W, 1))
    field[3][1][0] = 1

    # --- space
    new_space = processor.remap_observation_space(env.observation_space, env, DummyRLConfig())
    assert new_space == BoxSpace((env_org.H, env_org.W, 1), 0, 1, np.uint8, SpaceTypes.IMAGE)

    # --- decode
    new_state = processor.remap_observation([1, 3], None, env)
    assert isinstance(new_state, np.ndarray)
    assert (new_state == field).all()


def test_calc_action_values():
    env = cast(grid.Grid, srl.make_env("Grid").unwrapped)

    V = env.value_iteration()
    Q = env.calc_action_values()
    env.print_state_values(V)
    env.print_action_values(Q)
