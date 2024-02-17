from typing import List, Tuple

import srl
from srl.base.define import EnvActionType, EnvObservationType, EnvObservationTypes, InfoType, RenderModes, RLActionType
from srl.base.env import registration as env_registration
from srl.base.env.base import EnvBase
from srl.base.rl.base import RLWorker
from srl.base.rl.config import DummyRLConfig
from srl.base.rl.registration import register as rl_register
from srl.base.rl.worker_run import WorkerRun
from srl.base.run.context import RunContext
from srl.base.run.core_play import play
from srl.base.spaces.discrete import DiscreteSpace
from srl.base.spaces.space import SpaceBase


class StubEnv(EnvBase):
    @property
    def action_space(self) -> SpaceBase:
        return DiscreteSpace(5)

    @property
    def observation_space(self) -> SpaceBase:
        return DiscreteSpace(5)

    @property
    def observation_type(self) -> EnvObservationTypes:
        return EnvObservationTypes.DISCRETE

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

    def reset(self) -> Tuple[EnvObservationType, InfoType]:
        return 1, {}

    def step(self, action: EnvActionType) -> Tuple[EnvObservationType, List[float], bool, InfoType]:
        return 2, [10.0], False, {}


env_registration.register("StubEnvCore", entry_point=__name__ + ":StubEnv")


class StubWorker(RLWorker):
    def on_reset(self, worker: WorkerRun) -> InfoType:
        assert worker.prev_state == [0]
        assert worker.state == [1]
        assert worker.prev_action == 0
        assert worker.prev_invalid_actions == []
        assert worker.invalid_actions == []
        self.step_c = 0
        return {}

    def policy(self, worker: WorkerRun) -> Tuple[RLActionType, InfoType]:
        if self.step_c == 0:
            assert worker.prev_state == [0]
            assert worker.state == [1]
            assert worker.prev_action == 0
            assert worker.reward == 0
            assert not worker.done
            assert worker.prev_invalid_actions == []
            assert worker.invalid_actions == []
        else:
            assert worker.prev_state == [1]
            assert worker.state == [2]
            assert worker.prev_action == 2
            assert worker.reward == 10
            assert not worker.done
            assert worker.prev_invalid_actions == []
            assert worker.invalid_actions == []
        return 2, {}

    def on_step(self, worker: WorkerRun) -> InfoType:
        if self.step_c == 0:
            assert worker.prev_state == [1]
            assert worker.state == [2]
            assert worker.prev_action == 2
            assert worker.reward == 10
            assert not worker.done
            assert worker.prev_invalid_actions == []
            assert worker.invalid_actions == []
        else:
            assert worker.prev_state == [2]
            assert worker.state == [2]
            assert worker.prev_action == 2
            assert worker.reward == 10
            assert not worker.done
            assert worker.prev_invalid_actions == []
            assert worker.invalid_actions == []
        self.step_c += 1
        return {}

    def render_terminal(self, worker: WorkerRun, **kwargs) -> None:
        if self.step_c == 0:
            assert worker.prev_state == [0]
            assert worker.state == [1]
            assert worker.prev_action == 2
            assert worker.reward == 0
            assert not worker.done
            assert worker.prev_invalid_actions == []
            assert worker.invalid_actions == []
        else:
            assert worker.prev_state == [1]
            assert worker.state == [2]
            assert worker.prev_action == 2
            assert worker.reward == 10
            assert not worker.done
            assert worker.prev_invalid_actions == []
            assert worker.invalid_actions == []


rl_register(
    DummyRLConfig(name="StubWorker"),
    memory_entry_point="dummy",
    parameter_entry_point="dummy",
    trainer_entry_point="dummy",
    worker_entry_point=__name__ + ":StubWorker",
)


def test_play_worker():

    env_config = srl.EnvConfig("StubEnvCore", enable_assertion_value=True)
    rl_config = DummyRLConfig(name="StubWorker")

    context = RunContext(
        env_config,
        rl_config,
        max_episodes=1,
        max_steps=2,
        render_mode=RenderModes.terminal,
    )
    env = srl.make_env(env_config)
    parameter = srl.make_parameter(rl_config, env)
    memory = srl.make_memory(rl_config, env)
    trainer = srl.make_trainer(rl_config, parameter, memory)
    worker = srl.make_worker(rl_config, env, parameter, memory)
    play(context, env, parameter, memory, trainer, [worker])
