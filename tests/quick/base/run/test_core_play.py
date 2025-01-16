import pytest

import srl
from srl.base.context import RunContext
from srl.base.define import RenderModes, RLActionType, SpaceTypes
from srl.base.env import registration as env_registration
from srl.base.env.base import EnvBase
from srl.base.rl.config import DummyRLConfig
from srl.base.rl.registration import register as rl_register
from srl.base.rl.worker import RLWorker
from srl.base.run.play import play
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.space import SpaceBase
from srl.utils import common


class StubEnv(EnvBase):
    @property
    def action_space(self) -> SpaceBase:
        return DiscreteSpace(5)

    @property
    def observation_space(self) -> SpaceBase:
        return DiscreteSpace(5)

    @property
    def observation_type(self) -> SpaceTypes:
        return SpaceTypes.DISCRETE

    # --- properties
    @property
    def max_episode_steps(self) -> int:
        return 2

    @property
    def player_num(self) -> int:
        return 1

    @property
    def next_player_index(self) -> int:
        return 0

    def reset(self, **kwargs):
        return 1

    def step(self, action):
        return 2, [10.0], False, False


class StubWorker(RLWorker):
    def on_reset(self, worker):
        assert worker.prev_state == 0
        assert worker.state == 1
        assert worker.action == 0
        assert worker.prev_invalid_actions == []
        assert worker.invalid_actions == []
        self.step_c = 0

    def policy(self, worker) -> RLActionType:
        if self.step_c == 0:
            assert worker.prev_state == 0
            assert worker.state == 1
            assert worker.action == 0
            assert worker.reward == 0
            assert not worker.done
            assert worker.prev_invalid_actions == []
            assert worker.invalid_actions == []
        else:
            assert worker.prev_state == 1
            assert worker.state == 2
            assert worker.action == 2
            assert worker.reward == 10
            assert not worker.done
            assert worker.prev_invalid_actions == []
            assert worker.invalid_actions == []
        return 2

    def on_step(self, worker):
        if self.step_c == 0:
            assert worker.prev_state == 1
            assert worker.state == 2
            assert worker.action == 2
            assert worker.reward == 10
            assert not worker.done
            assert worker.prev_invalid_actions == []
            assert worker.invalid_actions == []
        else:
            assert worker.prev_state == 2
            assert worker.state == 2
            assert worker.action == 2
            assert worker.reward == 10
            assert not worker.done
            assert worker.prev_invalid_actions == []
            assert worker.invalid_actions == []
        self.step_c += 1

    def render_terminal(self, worker, **kwargs) -> None:
        if self.step_c == 0:
            assert worker.prev_state == 0
            assert worker.state == 1
            assert worker.action == 2
            assert worker.reward == 0
            assert not worker.done
            assert worker.prev_invalid_actions == []
            assert worker.invalid_actions == []
        else:
            assert worker.prev_state == 1
            assert worker.state == 2
            assert worker.action == 2
            assert worker.reward == 10
            assert not worker.done
            assert worker.prev_invalid_actions == []
            assert worker.invalid_actions == []


@pytest.fixture(scope="function", autouse=True)
def scope_function():
    env_registration.register(
        "StubEnvCore",
        entry_point=__name__ + ":StubEnv",
        check_duplicate=False,
    )
    rl_register(
        DummyRLConfig(name="StubWorker"),
        memory_entry_point="",
        parameter_entry_point="",
        trainer_entry_point="",
        worker_entry_point=__name__ + ":StubWorker",
        check_duplicate=False,
    )
    yield


def test_play_worker():
    common.logger_print()

    env_config = srl.EnvConfig("StubEnvCore", enable_assertion=True)
    rl_config = DummyRLConfig(name="StubWorker")

    context = RunContext(
        max_episodes=1,
        max_steps=2,
        render_mode=RenderModes.terminal,
    )
    env = env_config.make()
    parameter = rl_config.make_parameter(env)
    memory = rl_config.make_memory(env)
    trainer = rl_config.make_trainer(parameter, memory)
    worker = rl_config.make_worker(env, parameter, memory)
    play(context, env, [worker], 0, trainer)
