from typing import cast

import pytest

import srl
from srl.base.context import RunContext
from srl.base.env.registration import register as register_env
from srl.base.rl.registration import register as register_rl
from srl.base.rl.worker import RLWorker
from tests.quick.base.rl import worker_run_stub


class WorkerRunStubRLWorker(RLWorker):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.tracking_size = 0

    def on_reset(self, worker):
        for _ in range(2):
            worker.add_tracking({"n": 2})

        n = worker.get_tracking("n")
        assert n == [2, 2]
        aaa = worker.get_tracking("aaa")
        assert aaa == [None, None]

    def policy(self, worker):
        if worker.step_in_episode == 0:
            n = worker.get_tracking("n")
            assert n == [2, 2]
            aaa = worker.get_tracking("aaa")
            assert aaa == [None, None]
        return 1

    def on_step(self, worker):
        worker.add_tracking({"n": 3})

        if worker.step_in_episode == 1:
            n = worker.get_tracking("n")
            assert n == [2, 2, 3]
            aaa = worker.get_tracking("aaa")
            assert aaa == [None, None, None]

        if worker.done:
            for _ in range(2):
                worker.add_tracking({"n": 4})
            n = worker.get_tracking("n")
            assert n == [2, 2, 3, 3, 4, 4]
            aaa = worker.get_tracking("aaa")
            assert len(aaa) == 6


@pytest.fixture(scope="function", autouse=True)
def scope_function():
    register_env(id="Stub", entry_point=worker_run_stub.__name__ + ":WorkerRunStubEnv", check_duplicate=False)
    register_rl(worker_run_stub.WorkerRunStubRLConfig(), "", "", "", __name__ + ":WorkerRunStubRLWorker", check_duplicate=False)
    yield


# ------------------------------------------------------------------


def test_episode():
    env = srl.make_env(srl.EnvConfig("Stub"))
    env_org = cast(worker_run_stub.WorkerRunStubEnv, env.unwrapped)
    env_org.s_states = [1, 2, 3]  # 2step

    rl_config = worker_run_stub.WorkerRunStubRLConfig()
    rl_config.enable_assertion = True
    rl_config._use_render_image_state = True

    # --- setup
    rl_config.setup(env)
    worker = srl.make_worker(rl_config, env)
    worker_base = cast(WorkerRunStubRLWorker, worker.worker)

    context = RunContext()
    env.setup(context)
    worker.setup(context)

    # --- reset
    env.reset()
    worker.reset(0)

    # 1episode
    while not env.done:
        act = worker.policy()
        env.step(act)
        worker.on_step()

    env.teardown()
    worker.teardown()
