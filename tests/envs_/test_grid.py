from typing import cast

import numpy as np

import srl
from srl.base.define import EnvObservationTypes
from srl.base.spaces.box import BoxSpace  # noqa F401
from srl.envs import grid
from srl.test import TestEnv
from srl.test.processor import TestProcessor


def test_grid():
    tester = TestEnv()
    tester.play_test("Grid")


def test_easy_grid():
    tester = TestEnv()
    tester.play_test("EasyGrid")


def test_processor():
    tester = TestProcessor()
    processor = grid.LayerProcessor()
    env_name = "Grid"

    env = grid.Grid()
    field = np.zeros((env.H, env.W, 1))
    field[3][1][0] = 1

    tester.run(processor, env_name)
    tester.preprocess_observation_space(
        processor,
        env_name,
        after_type=EnvObservationTypes.IMAGE,
        after_space=BoxSpace((env.H, env.W, 1), 0, 1),
    )
    tester.preprocess_observation(
        processor,
        env_name,
        in_observation=[1, 3],
        out_observation=field,
    )


def test_calc_action_values():
    env = cast(grid.Grid, srl.make_env("Grid").unwrapped)

    V = env.value_iteration()
    Q = env.calc_action_values()
    env.print_state_values(V)
    env.print_action_values(Q)
