from typing import cast

import numpy as np
import pytest

import srl
from srl.base.context import RunContext
from srl.base.env.registration import register as register_env
from srl.base.rl.registration import register as register_rl
from srl.base.rl.worker import RLWorker
from srl.utils.common import is_package_installed
from tests.quick.base.rl import worker_run_stub


@pytest.fixture(scope="function", autouse=True)
def scope_function():
    register_env(id="Stub", kwargs={"invalid_action": True}, entry_point=worker_run_stub.__name__ + ":WorkerRunStubEnv", check_duplicate=False)
    yield


# ------------------------------------------------------------------


class WorkerRunStubEpisode(RLWorker):
    def on_reset(self, worker):
        assert worker.state == [1]
        if self.config.use_render_image_state():
            assert np.mean(worker.render_image_state) == 1
        assert worker.invalid_actions == [1]
        self.n_render = 0

    def policy(self, worker):
        if self.step_in_episode == 0:
            assert worker.state == [1]
            if self.config.use_render_image_state():
                assert np.mean(worker.render_image_state) == 1
            assert worker.invalid_actions == [1]
            return 1
        elif self.step_in_episode == 1:
            assert worker.state == [2]
            if self.config.use_render_image_state():
                assert np.mean(worker.render_image_state) == 2
            assert worker.invalid_actions == [2]
            return 2
        assert False

    def on_step(self, worker):
        if self.step_in_episode == 1:
            assert worker.state == [1]
            assert worker.next_state == [2]
            if self.config.use_render_image_state():
                assert np.mean(worker.render_image_state) == 1
                assert np.mean(worker.next_render_image_state) == 2
            assert worker.action == 1
            assert worker.reward == 1.0
            assert worker.done == False  # noqa: E712
            assert worker.terminated == False  # noqa: E712
            assert worker.invalid_actions == [1]
            assert worker.next_invalid_actions == [2]
        elif self.step_in_episode == 2:
            assert worker.state == [2]
            assert worker.next_state == [3]
            if self.config.use_render_image_state():
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
        self.n_render += 1
        if self.step_in_episode == 0:
            assert worker.prev_state == [0]
            assert worker.state == [1]
            assert worker.action == 1
            if self.config.use_render_image_state():
                assert np.mean(worker.prev_render_image_state) == 0
                assert np.mean(worker.render_image_state) == 1
            assert worker.prev_invalid_actions == []
            assert worker.invalid_actions == [1]
        elif self.step_in_episode == 1:
            assert worker.prev_state == [1]
            assert worker.state == [2]
            assert worker.action == 2
            if self.config.use_render_image_state():
                assert np.mean(worker.prev_render_image_state) == 1
                assert np.mean(worker.render_image_state) == 2
            assert worker.prev_invalid_actions == [1]
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
    rl_config._use_render_image_state = is_package_installed("pygame")

    # --- setup
    rl_config.setup(env)
    worker = srl.make_worker(rl_config, env)

    context = RunContext()
    env.setup(context)
    worker.setup(context, render_mode="terminal")

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

    assert cast(WorkerRunStubEpisode, worker.worker).n_render > 0


def test_episode_runner():
    register_rl(worker_run_stub.WorkerRunStubRLConfig("StubEpisode"), "", "", "", __name__ + ":WorkerRunStubEpisode", check_duplicate=False)
    rl_config = worker_run_stub.WorkerRunStubRLConfig("StubEpisode")
    rl_config.enable_assertion = True
    rl_config._use_render_image_state = is_package_installed("pygame")

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
        if self.config.use_render_image_state():
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
        assert worker.invalid_actions == [1]
        self.n_render = 0

    def policy(self, worker):
        if self.step_in_episode == 0:
            assert worker.state == [0, 1]
            assert worker.get_state_one_step(-2) == [0]
            assert worker.get_state_one_step(-1) == [1]
            if self.config.use_render_image_state():
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
            assert worker.invalid_actions == [1]
            return 1
        elif self.step_in_episode == 1:
            assert worker.state == [1, 2]
            assert worker.get_state_one_step(-2) == [1]
            assert worker.get_state_one_step(-1) == [2]
            if self.config.use_render_image_state():
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
        if self.step_in_episode == 1:
            assert worker.state == [0, 1]
            assert worker.next_state == [1, 2]
            if self.config.use_render_image_state():
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
        elif self.step_in_episode == 2:
            assert worker.state == [1, 2]
            assert worker.next_state == [2, 3]
            if self.config.use_render_image_state():
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
        self.n_render += 1
        if self.step_in_episode == 0:
            assert worker.prev_state == [0, 0]
            assert worker.state == [0, 1]
            assert worker.get_state_one_step(-2) == [0]
            assert worker.get_state_one_step(-1) == [1]
            if self.config.use_render_image_state():
                assert (
                    worker.prev_render_image_state
                    == np.stack(
                        [
                            np.full((64, 32, 3), 0),
                            np.full((64, 32, 3), 0),
                        ],
                        axis=0,
                    )
                ).all()
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
            assert worker.prev_invalid_actions == []
            assert worker.invalid_actions == [1]
        elif self.step_in_episode == 1:
            assert worker.prev_state == [0, 1]
            assert worker.state == [1, 2]
            assert worker.get_state_one_step(-2) == [1]
            assert worker.get_state_one_step(-1) == [2]
            if self.config.use_render_image_state():
                assert (
                    worker.prev_render_image_state
                    == np.stack(
                        [
                            np.full((64, 32, 3), 0),
                            np.full((64, 32, 3), 1),
                        ],
                        axis=0,
                    )
                ).all()
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
            assert worker.prev_invalid_actions == [1]
            assert worker.invalid_actions == [2]
        elif self.step_in_episode == 2:
            pass
        else:
            assert False


def test_episode_stacked():
    env = srl.make_env(srl.EnvConfig("Stub"))
    env_org = cast(worker_run_stub.WorkerRunStubEnv, env.unwrapped)
    env_org.s_states = [1, 2, 3]  # 2step

    register_rl(worker_run_stub.WorkerRunStubRLConfig("StubEpisodeStacked"), "", "", "", __name__ + ":WorkerRunStubEpisodeStacked", check_duplicate=False)
    rl_config = worker_run_stub.WorkerRunStubRLConfig("StubEpisodeStacked")
    rl_config.enable_assertion = True
    rl_config._use_render_image_state = is_package_installed("pygame")
    rl_config.window_length = 2
    rl_config.render_image_window_length = 2

    # --- setup
    rl_config.setup(env)
    worker = srl.make_worker(rl_config, env)

    context = RunContext(env_render_mode="terminal", rl_render_mode="terminal")
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

    assert cast(WorkerRunStubEpisodeStacked, worker.worker).n_render > 0


def test_episode_stacked_runner():
    register_rl(worker_run_stub.WorkerRunStubRLConfig("StubEpisodeStacked"), "", "", "", __name__ + ":WorkerRunStubEpisodeStacked", check_duplicate=False)
    rl_config = worker_run_stub.WorkerRunStubRLConfig("StubEpisodeStacked")
    rl_config.enable_assertion = True
    rl_config._use_render_image_state = is_package_installed("pygame")
    rl_config.window_length = 2
    rl_config.render_image_window_length = 2

    import srl

    runner = srl.Runner("Stub", rl_config)
    env = runner.make_env()
    env_org = cast(worker_run_stub.WorkerRunStubEnv, env.unwrapped)
    env_org.s_states = [1, 2, 3]  # 2step

    runner.render_terminal()

    assert cast(WorkerRunStubEpisodeStacked, runner.worker.worker).n_render > 0
