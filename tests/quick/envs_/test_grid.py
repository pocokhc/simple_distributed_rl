from typing import cast

import numpy as np

import srl
from srl.base.define import SpaceTypes
from srl.base.spaces.box import BoxSpace
from srl.envs import grid
from srl.test.env import env_test


def test_grid():
    env_test("Grid")


def test_easy_grid():
    env_test("EasyGrid")


def test_processor():
    processor = grid.LayerProcessor()
    env = srl.make_env("Grid")
    env.setup()
    env.reset()

    env_org: grid.Grid = env.unwrapped
    field = np.zeros((env_org.H, env_org.W, 1))
    field[3][1][0] = 1

    # --- space
    new_space = processor.remap_observation_space(env.observation_space, env)
    assert new_space == BoxSpace((env_org.H, env_org.W, 1), 0, 1, np.uint8, SpaceTypes.FEATURE_MAP)

    # --- decode
    new_state = processor.remap_observation(None, env.observation_space, new_space, env_run=env)
    assert isinstance(new_state, np.ndarray)
    assert (new_state == field).all()


def test_calc_action_values():
    env = grid.Grid()

    V = env.calc_state_values()
    Q = env.calc_action_values()
    env.print_state_values(V)
    env.print_action_values(Q)

    print(env.prediction_reward(Q))
