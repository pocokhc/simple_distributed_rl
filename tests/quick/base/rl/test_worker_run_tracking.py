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

    def on_setup(self, worker, context: RunContext) -> None:
        worker.enable_tracking()

    def on_reset(self, worker):
        state = worker.get_tracking("state")
        assert state == [[1]]
        render_image = worker.get_tracking("render_image")
        assert len(render_image) == 1
        act = worker.get_tracking("action")
        assert act == []
        invalid_actions = worker.get_tracking("invalid_actions")
        assert invalid_actions == []
        reward = worker.get_tracking("reward")
        assert reward == []
        terminated = worker.get_tracking("terminated")
        assert terminated == []

        # --- add dummy
        for _ in range(2):
            worker.add_tracking_dummy_step(
                tracking_data={"n": 2},
                is_reset=True,
            )

        state = worker.get_tracking("state")
        assert state == [[0], [0], [1]]
        render_image = worker.get_tracking("render_image")
        assert len(render_image) == 3
        act = worker.get_tracking("action")
        assert act == [0, 0]
        invalid_actions = worker.get_tracking("invalid_actions")
        assert invalid_actions == [[], []]
        reward = worker.get_tracking("reward")
        assert reward == [0.0, 0.0]
        terminated = worker.get_tracking("terminated")
        assert terminated == [0, 0]
        n = worker.get_tracking_data("n")
        assert n == [2, 2]
        aaa = worker.get_tracking_data("aaa")
        assert aaa == [None, None]

    def policy(self, worker):
        if worker.total_step == 0:
            state = worker.get_tracking("state")
            assert state == [[0], [0], [1]]
            render_image = worker.get_tracking("render_image")
            assert len(render_image) == 3
            act = worker.get_tracking("action")
            assert act == [0, 0]
            invalid_actions = worker.get_tracking("invalid_actions")
            assert invalid_actions == [[], []]
            reward = worker.get_tracking("reward")
            assert reward == [0.0, 0.0]
            terminated = worker.get_tracking("terminated")
            assert terminated == [0, 0]
            n = worker.get_tracking_data("n")
            assert n == [2, 2]
            aaa = worker.get_tracking_data("aaa")
            assert aaa == [None, None]

        return 1

    def on_step(self, worker):
        worker.add_tracking_data({"n": 3})

        if worker.total_step == 1:
            state = worker.get_tracking("state")
            assert state == [[0], [0], [1], [2]]
            render_image = worker.get_tracking("render_image")
            assert len(render_image) == 4
            act = worker.get_tracking("action")
            assert act == [0, 0, 1]
            invalid_actions = worker.get_tracking("invalid_actions")
            assert invalid_actions == [[], [], []]
            reward = worker.get_tracking("reward")
            assert reward == [0.0, 0.0, 1.0]
            terminated = worker.get_tracking("terminated")
            assert terminated == [0, 0, 0]
            n = worker.get_tracking_data("n")
            assert n == [2, 2, 3]
            aaa = worker.get_tracking_data("aaa")
            assert aaa == [None, None, None]

        if worker.done:
            for _ in range(2):
                worker.add_tracking_dummy_step(tracking_data={"n": 4})

            state = worker.get_tracking("state")
            assert state == [[0], [0], [1], [2], [3], [0], [0]]
            render_image = worker.get_tracking("render_image")
            assert len(render_image) == 7
            act = worker.get_tracking("action")
            assert act == [0, 0, 1, 1, 0, 0]
            invalid_actions = worker.get_tracking("invalid_actions")
            assert len(invalid_actions) == 6
            reward = worker.get_tracking("reward")
            assert reward == [0.0, 0.0, 1.0, 2.0, 0.0, 0.0]
            terminated = worker.get_tracking("terminated")
            assert terminated == [0, 0, 0, 1, 0, 0]
            n = worker.get_tracking_data("n")
            assert n == [2, 2, 3, 3, 4, 4]
            aaa = worker.get_tracking_data("aaa")
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
