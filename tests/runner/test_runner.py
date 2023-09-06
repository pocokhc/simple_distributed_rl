import json
from pprint import pprint
from typing import cast

import numpy as np

from srl.algorithms import dqn, ql
from srl.envs import grid
from srl.runner.runner import Runner
from srl.utils import common

common.logger_print()


class NotJsonClass:
    def __init__(self):
        self.a = 1


def test_config_to_dict():
    runner = Runner("Grid", ql.Config())

    runner.config.players = [
        None,
        "AAA",
        ("aa", {"bb": "cc"}),
        dqn.Config(),
    ]

    def _dummy():
        return 1

    runner.config.env_config.gym_make_func = _dummy  # type: ignore
    runner.config.env_config.kwargs = {"a": 1, "not_json_class": NotJsonClass()}

    c1 = runner.config
    json_dict = c1.to_dict()
    pprint(json_dict)
    json.dumps(json_dict)


def test_context_to_dict():
    runner = Runner("Grid", ql.Config())
    c1 = runner.context

    json_dict = c1.to_dict()
    pprint(json_dict)
    json.dumps(json_dict)


def test_get_env_init_state():
    rl_config = ql.Config()
    rl_config.processors.append(grid.LayerProcessor())
    runner = Runner("Grid", rl_config)

    env_state = runner.get_env_init_state(encode=False)
    print(env_state)
    assert env_state == [1, 3]

    rl_state = runner.get_env_init_state(encode=True)
    rl_state = cast(np.ndarray, rl_state)
    print(rl_state)
    assert rl_state.shape == (1, 5, 6)
    assert rl_state[0][3][1] == 1
