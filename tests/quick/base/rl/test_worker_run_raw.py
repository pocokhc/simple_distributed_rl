from typing import cast

import numpy as np
import pytest

import srl
from srl.base.context import RunContext
from srl.base.env.registration import register as register_env
from srl.base.rl.registration import register as register_rl
from srl.base.rl.worker import RLWorker
from tests.quick.base.rl import worker_run_stub


@pytest.fixture(scope="function", autouse=True)
def scope_function():
    register_env(id="Stub", kwargs={"invalid_action": True}, entry_point=worker_run_stub.__name__ + ":WorkerRunStubEnv", check_duplicate=False)
    yield


# ------------------------------------------------------------------


class WorkerRunStubEpisode(RLWorker):
    def on_reset(self, worker):
        assert worker.state == [1]
        assert np.mean(worker.render_image_state) == 1
        assert worker.invalid_actions == [1]

    def policy(self, worker):
        if self.total_step == 0:
            assert worker.state == [1]
            assert np.mean(worker.render_image_state) == 1
            assert worker.invalid_actions == [1]
            return 1
        elif self.total_step == 1:
            assert worker.state == [2]
            assert np.mean(worker.render_image_state) == 2
            assert worker.invalid_actions == [2]
            return 2
        assert False

    def on_step(self, worker):
        if self.total_step == 0:
            assert worker.state == [1]
            assert worker.next_state == [2]
            assert np.mean(worker.render_image_state) == 1
            assert np.mean(worker.next_render_image_state) == 2
            assert worker.action == 1
            assert worker.reward == 1.0
            assert worker.done == False  # noqa: E712
            assert worker.terminated == False  # noqa: E712
            assert worker.invalid_actions == [1]
            assert worker.next_invalid_actions == [2]
        elif self.total_step == 1:
            assert worker.state == [2]
            assert worker.next_state == [3]
            assert np.mean(worker.render_image_state) == 2
            assert np.mean(worker.next_render_image_state) == 3
            assert worker.action == 2
            assert worker.reward == 2.0
            assert worker.done == True  # noqa: E712
            assert worker.terminated == True  # noqa: E712
            assert worker.invalid_actions == [2]
            assert worker.next_invalid_actions == [3]
        else:
            assert False

    def render_terminal(self, worker, **kwargs) -> None:
        if self.total_step == 0:
            assert worker.state == [1]
            assert np.mean(worker.render_image_state) == 1
            assert worker.invalid_actions == [1]
        elif self.total_step == 1:
            assert worker.state == [2]
            assert np.mean(worker.render_image_state) == 2
            assert worker.invalid_actions == [2]
        else:
            assert False


def test_episode():
    env = srl.make_env(srl.EnvConfig("Stub"))
    env_org = cast(worker_run_stub.WorkerRunStubEnv, env.unwrapped)
    env_org.s_states = [1, 2, 3]  # 2step

    register_rl(worker_run_stub.WorkerRunStubRLConfig("StubEpisode"), "", "", "", __name__ + ":WorkerRunStubEpisode", check_duplicate=False)
    rl_config = worker_run_stub.WorkerRunStubRLConfig("StubEpisode")
    rl_config.enable_assertion = True
    rl_config._use_render_image_state = True

    # --- setup
    rl_config.setup(env)
    worker = srl.make_worker(rl_config, env)

    context = RunContext(rendering=True)
    env.setup(context)
    worker.setup(context, render_mode="rgb_array")

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


def test_episode_runner():
    register_rl(worker_run_stub.WorkerRunStubRLConfig("StubEpisode"), "", "", "", __name__ + ":WorkerRunStubEpisode", check_duplicate=False)
    rl_config = worker_run_stub.WorkerRunStubRLConfig("StubEpisode")
    rl_config.enable_assertion = True
    rl_config._use_render_image_state = True

    import srl

    runner = srl.Runner("Stub", rl_config)
    env = runner.make_env()
    env_org = cast(worker_run_stub.WorkerRunStubEnv, env.unwrapped)
    env_org.s_states = [1, 2, 3]  # 2step

    runner.evaluate(max_episodes=1)


# ---------------------------------------------


class WorkerRunStubEpisodeStacked(RLWorker):
    def on_reset(self, worker):
        assert worker.state == [0, 1]
        assert worker.get_state_one_step(-2) == [0]
        assert worker.get_state_one_step(-1) == [1]
        assert (
            worker.render_image_state
            == np.stack(
                [
                    np.zeros((64, 32, 3)),
                    np.ones((64, 32, 3)),
                ],
                axis=0,
            )
        ).all()
        assert worker.invalid_actions == [1]

    def policy(self, worker):
        if self.total_step == 0:
            assert worker.state == [0, 1]
            assert worker.get_state_one_step(-2) == [0]
            assert worker.get_state_one_step(-1) == [1]
            assert (
                worker.render_image_state
                == np.stack(
                    [
                        np.zeros((64, 32, 3)),
                        np.ones((64, 32, 3)),
                    ],
                    axis=0,
                )
            ).all()
            assert worker.invalid_actions == [1]
            return 1
        elif self.total_step == 1:
            assert worker.state == [1, 2]
            assert worker.get_state_one_step(-2) == [1]
            assert worker.get_state_one_step(-1) == [2]
            assert (
                worker.render_image_state
                == np.stack(
                    [
                        np.full((64, 32, 3), 1),
                        np.full((64, 32, 3), 2),
                    ],
                    axis=0,
                )
            ).all()
            assert worker.invalid_actions == [2]
            return 2
        assert False

    def on_step(self, worker):
        if self.total_step == 0:
            assert worker.state == [0, 1]
            assert worker.next_state == [1, 2]
            assert (
                worker.render_image_state
                == np.stack(
                    [
                        np.full((64, 32, 3), 0),
                        np.full((64, 32, 3), 1),
                    ],
                    axis=0,
                )
            ).all()
            assert (
                worker.next_render_image_state
                == np.stack(
                    [
                        np.full((64, 32, 3), 1),
                        np.full((64, 32, 3), 2),
                    ],
                    axis=0,
                )
            ).all()
            assert worker.action == 1
            assert worker.reward == 1.0
            assert worker.done == False  # noqa: E712
            assert worker.terminated == False  # noqa: E712
            assert worker.invalid_actions == [1]
            assert worker.next_invalid_actions == [2]
        elif self.total_step == 1:
            assert worker.state == [1, 2]
            assert worker.next_state == [2, 3]
            assert (
                worker.render_image_state
                == np.stack(
                    [
                        np.full((64, 32, 3), 1),
                        np.full((64, 32, 3), 2),
                    ],
                    axis=0,
                )
            ).all()
            assert (
                worker.next_render_image_state
                == np.stack(
                    [
                        np.full((64, 32, 3), 2),
                        np.full((64, 32, 3), 3),
                    ],
                    axis=0,
                )
            ).all()
            assert worker.action == 2
            assert worker.reward == 2.0
            assert worker.done == True  # noqa: E712
            assert worker.terminated == True  # noqa: E712
            assert worker.invalid_actions == [2]
            assert worker.next_invalid_actions == [3]
        else:
            assert False

    def render_terminal(self, worker, **kwargs) -> None:
        if self.total_step == 0:
            assert worker.state == [0, 1]
            assert worker.get_state_one_step(-2) == [0]
            assert worker.get_state_one_step(-1) == [1]
            assert (
                worker.render_image_state
                == np.stack(
                    [
                        np.zeros((64, 32, 3)),
                        np.ones((64, 32, 3)),
                    ],
                    axis=0,
                )
            ).all()
            assert worker.invalid_actions == [1]
        elif self.total_step == 1:
            assert worker.state == [1, 2]
            assert worker.get_state_one_step(-2) == [1]
            assert worker.get_state_one_step(-1) == [2]
            assert (
                worker.render_image_state
                == np.stack(
                    [
                        np.full((64, 32, 3), 1),
                        np.full((64, 32, 3), 2),
                    ],
                    axis=0,
                )
            ).all()
            assert worker.invalid_actions == [2]
        else:
            assert False


def test_episode_stacked():
    env = srl.make_env(srl.EnvConfig("Stub"))
    env_org = cast(worker_run_stub.WorkerRunStubEnv, env.unwrapped)
    env_org.s_states = [1, 2, 3]  # 2step

    register_rl(worker_run_stub.WorkerRunStubRLConfig("StubEpisodeStacked"), "", "", "", __name__ + ":WorkerRunStubEpisodeStacked", check_duplicate=False)
    rl_config = worker_run_stub.WorkerRunStubRLConfig("StubEpisodeStacked")
    rl_config.enable_assertion = True
    rl_config._use_render_image_state = True
    rl_config.window_length = 2
    rl_config.render_image_window_length = 2

    # --- setup
    rl_config.setup(env)
    worker = srl.make_worker(rl_config, env)

    context = RunContext(rendering=True, render_mode="terminal")
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


def test_episode_stacked_runner():
    register_rl(worker_run_stub.WorkerRunStubRLConfig("StubEpisodeStacked"), "", "", "", __name__ + ":WorkerRunStubEpisodeStacked", check_duplicate=False)
    rl_config = worker_run_stub.WorkerRunStubRLConfig("StubEpisodeStacked")
    rl_config.enable_assertion = True
    rl_config._use_render_image_state = True
    rl_config.window_length = 2
    rl_config.render_image_window_length = 2

    import srl

    runner = srl.Runner("Stub", rl_config)
    env = runner.make_env()
    env_org = cast(worker_run_stub.WorkerRunStubEnv, env.unwrapped)
    env_org.s_states = [1, 2, 3]  # 2step

    runner.evaluate(max_episodes=1)
