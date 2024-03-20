import json
from pprint import pprint

import pytest

import srl
from srl.algorithms import dqn, ql
from srl.base.env import registration as env_registration
from srl.base.env.config import EnvConfig
from srl.base.rl import registration as rl_registration
from srl.base.rl.algorithms.env_worker import EnvWorker
from srl.base.rl.worker import RLWorker
from srl.base.run.context import RunContext
from srl.envs import grid
from srl.utils import common


class NotJsonClass:
    def __init__(self):
        self.a = 1


def test_to_dict():
    pytest.importorskip("tensorflow")
    common.logger_print()

    env_config = EnvConfig("Grid")
    rl_config = ql.Config()
    rl_config2 = dqn.Config()
    rl_config2.setup(srl.make_env(env_config))
    c = RunContext(env_config, rl_config)

    c.players = [
        None,
        "AAA",
        ("aa", {"bb": "cc"}),
        dqn.Config(),
        (dqn.Config(), srl.make_parameter(rl_config2).backup()),
    ]

    def _dummy():
        return 1

    c.env_config.gym_make_func = _dummy  # type: ignore
    c.env_config.kwargs = {"a": 1, "not_json_class": NotJsonClass()}

    json_dict = c.to_dict()
    pprint(json_dict)
    json.dumps(json_dict)


# --------------------------------


class StubEnvWorker(EnvWorker):
    def call_policy(self, env):
        pass


class StubEnv(grid.Grid):
    @property
    def player_num(self) -> int:
        return 8

    def make_worker(self, name: str, a=0):
        if name != "cpu":
            return None
        if a > 0:
            assert a == 1
        return StubEnvWorker()


env_registration.register("StubEnv", entry_point=__name__ + ":StubEnv")


class StubWorker(RLWorker):
    def __init__(self, b=0, **kwargs):
        super().__init__(**kwargs)
        if b > 0:
            assert b == 2

    def policy(self, worker):
        pass


rl_registration.register_rulebase("StubWorker", entry_point=__name__ + ":StubWorker")


def test_make_workers():
    env_config = EnvConfig("StubEnv")
    rl_config = ql.Config()
    env = srl.make_env(env_config)
    rl_config.setup(env)
    parameter = srl.make_parameter(rl_config)
    memory = srl.make_memory(rl_config)

    c = RunContext(env_config, rl_config)

    # --- non
    workers, main_worker_idx = c.make_workers(env, parameter, memory)
    assert len(workers) == 8
    assert main_worker_idx == 0
    assert workers[0].config.get_name() == "QL"
    for i in range(1, len(workers)):
        assert workers[i].config.get_name() == "random"

    # --- set
    other_rl = ql.Config()
    other_rl.setup(env)
    c.players = [
        "cpu",  # EnvWorker
        ("cpu", {"a": 1}),  # EnvWorker + args
        "StubWorker",  # RuleBase
        ("StubWorker", {"b": 2}),  # RuleBase + args
        other_rl,  # other rl
        (other_rl, srl.make_parameter(other_rl).backup()),  # other rl + param
        None,
        None,  # share param
        None,  # out of index
    ]
    workers, main_worker_idx = c.make_workers(env, parameter, memory)
    assert len(workers) == 8
    assert main_worker_idx == 6
    assert isinstance(workers[0].worker, StubEnvWorker)
    assert isinstance(workers[1].worker, StubEnvWorker)
    assert workers[0].worker != workers[1].worker
    assert isinstance(workers[2].worker, StubWorker)
    assert isinstance(workers[3].worker, StubWorker)
    assert workers[2].worker != workers[3].worker
    for i in [4, 5, 6, 7]:
        assert workers[i].config.get_name() == "QL"
        assert workers[i].worker.memory is not None
        assert workers[i].worker.parameter is not None
    # other rl
    assert workers[4].worker.memory is not workers[5].worker.memory
    assert workers[5].worker.memory is not workers[6].worker.memory
    assert workers[4].worker.parameter is not workers[5].worker.parameter
    assert workers[5].worker.parameter is not workers[6].worker.parameter
    # None
    assert workers[6].worker.memory is workers[7].worker.memory
    assert workers[6].worker.parameter is workers[7].worker.parameter
