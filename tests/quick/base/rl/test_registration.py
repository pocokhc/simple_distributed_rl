import pytest

import srl
from srl.algorithms import ql
from srl.base.env import registration as env_registration
from srl.base.rl import registration as rl_registration
from srl.base.rl.algorithms.env_worker import EnvWorker
from srl.base.rl.worker import RLWorker
from srl.envs import grid


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


class StubWorker(RLWorker):
    def policy(self, worker):
        pass


@pytest.fixture(scope="function", autouse=True)
def scope_function():
    env_registration.register("StubEnv", entry_point=__name__ + ":StubEnv", enable_assert=False)
    rl_registration.register_rulebase("StubWorker", entry_point=__name__ + ":StubWorker", enable_assert=False)
    yield


def test_make_workers():
    env_config = srl.EnvConfig("StubEnv")
    rl_config = ql.Config()
    env = env_config.make()
    rl_config.setup(env)
    parameter = rl_config.make_parameter()
    memory = rl_config.make_memory()

    # --- non
    workers, main_worker_idx = srl.make_workers([], env, rl_config, parameter, memory)
    assert len(workers) == 8
    assert main_worker_idx == 0
    assert workers[0].config.get_name() == "QL"
    for i in range(1, len(workers)):
        assert workers[i].config.get_name() == "random"

    # --- set
    other_rl = ql.Config()
    other_rl.setup(env)
    players = [
        "cpu",  # EnvWorker
        ("cpu", {"a": 1}),  # EnvWorker + args
        "StubWorker",  # RuleBase
        other_rl,  # other rl
        (other_rl, other_rl.make_parameter().backup()),  # other rl + param
        None,
        None,  # share param
        None,  # out of index
    ]
    workers, main_worker_idx = srl.make_workers(players, env, rl_config, parameter, memory)
    assert len(workers) == 8
    assert main_worker_idx == 5
    assert isinstance(workers[0].worker, StubEnvWorker)
    assert isinstance(workers[1].worker, StubEnvWorker)
    assert workers[0].worker != workers[1].worker
    assert isinstance(workers[2].worker, StubWorker)
    assert workers[3].worker != workers[4].worker
    for i in [3, 4, 5, 6]:
        assert workers[i].config.get_name() == "QL"
        assert workers[i].worker.memory is not None
        assert workers[i].worker.parameter is not None
    # other rl
    assert workers[3].worker.memory is not workers[4].worker.memory
    assert workers[4].worker.memory is not workers[5].worker.memory
    assert workers[3].worker.parameter is not workers[4].worker.parameter
    assert workers[4].worker.parameter is not workers[5].worker.parameter
    # None
    assert workers[5].worker.memory is workers[6].worker.memory
    assert workers[5].worker.parameter is workers[6].worker.parameter
