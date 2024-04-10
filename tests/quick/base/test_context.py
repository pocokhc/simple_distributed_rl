import json
from pprint import pprint

import pytest

import srl
from srl.algorithms import dqn
from srl.base.context import RunContext
from srl.base.env.config import EnvConfig
from srl.utils import common


class NotJsonClass:
    def __init__(self):
        self.a = 1


@pytest.mark.parametrize("framework", ["tensorflow", "torch"])
def test_to_dict(framework):
    if framework == "tensorflow":
        pytest.importorskip("tensorflow")
    elif framework == "torch":
        pytest.importorskip("torch")
    common.logger_print()

    env_config = EnvConfig("Grid")
    rl_config = dqn.Config()
    if framework == "tensorflow":
        rl_config.set_tensorflow()
    elif framework == "torch":
        rl_config.set_torch()

    rl_config.setup(srl.make_env(env_config))
    c = RunContext()

    c.players = [
        None,
        "AAA",
        ("aa", {"bb": "cc"}),
        dqn.Config(),
        (dqn.Config(), srl.make_parameter(rl_config).backup()),
    ]
    if framework == "tensorflow":
        c.players[3].set_tensorflow()
    elif framework == "torch":
        c.players[3].set_torch()

    def _dummy():
        return 1

    env_config.gym_make_func = _dummy  # type: ignore
    env_config.kwargs = {"a": 1, "not_json_class": NotJsonClass()}

    json_dict = c.to_dict()
    pprint(json_dict)
    json.dumps(json_dict)
