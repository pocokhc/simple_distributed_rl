from typing import Optional

import pytest
import pytest_mock
import pytest_timeout  # noqa F401

import srl
from srl.algorithms import ql, ql_agent57
from srl.base.context import RunContext
from srl.base.run.callback import RunCallback, TrainerCallback
from srl.base.run.core_play import RunStateActor, play
from srl.base.run.core_train_only import RunStateTrainer, play_trainer_only


class DummyCallback(RunCallback):
    def on_episode_begin(self, context: RunContext, state: RunStateActor) -> None:
        pass  # do nothing

    def on_episode_end(self, context: RunContext, state: RunStateActor) -> None:
        pass  # do nothing

    def on_step_action_before(self, context: RunContext, state: RunStateActor) -> None:
        pass  # do nothing

    def on_step_action_after(self, context: RunContext, state: RunStateActor) -> None:
        pass  # do nothing

    def on_step_begin(self, context: RunContext, state: RunStateActor) -> None:
        pass  # do nothing

    def on_step_end(self, context: RunContext, state: RunStateActor) -> Optional[bool]:
        return False

    def on_skip_step(self, context: RunContext, state: RunStateActor) -> None:
        pass  # do nothing


def test_callback(mocker: pytest_mock.MockerFixture):
    env_config = srl.EnvConfig("Grid", frameskip=4)
    rl_config = ql.Config()

    env = env_config.make()
    parameter = rl_config.make_parameter(env)
    memory = rl_config.make_memory(env)
    trainer = rl_config.make_trainer(parameter, memory)
    workers, main_worker_idx = rl_config.make_workers([], env, parameter, memory)
    c = mocker.Mock(spec=DummyCallback)

    context = RunContext()
    context.training = True
    context.max_episodes = 1
    state = play(context, env, workers, main_worker_idx, trainer=trainer, callbacks=[c])

    assert state.total_step >= 1
    assert state.episode_count == 1
    assert c.on_episodes_begin.call_count == 1
    assert c.on_episodes_end.call_count == 1
    assert c.on_episode_begin.call_count >= c.on_episode_end.call_count  # episode中に終了する可能性あり
    assert c.on_episode_end.call_count == 1
    assert c.on_step_begin.call_count >= 1
    assert c.on_step_begin.call_count == c.on_step_action_before.call_count
    assert c.on_step_begin.call_count == c.on_step_action_after.call_count
    assert c.on_step_begin.call_count == c.on_step_end.call_count
    assert c.on_skip_step.call_count >= 1  # episode終了タイミングで変化する


class DummyTrainerCallback(TrainerCallback):
    def on_train_before(self, context: RunContext, state: RunStateTrainer):
        pass

    def on_train_after(self, context: RunContext, state: RunStateTrainer) -> Optional[bool]:
        return False


@pytest.mark.timeout(2)  # pip install pytest_timeout
def test_trainer_callback(mocker: pytest_mock.MockerFixture):
    env_config = srl.EnvConfig("Grid")
    rl_config = ql_agent57.Config()
    rl_config.memory.warmup_size = 100
    rl_config.batch_size = 1

    env = env_config.make()
    parameter = rl_config.make_parameter(env)
    memory = rl_config.make_memory(env)
    workers, main_worker_idx = rl_config.make_workers([], env, parameter, memory)

    context = RunContext()
    context.training = True
    context.max_memory = 100
    context.disable_trainer = True
    play(context, env, workers, main_worker_idx)
    assert memory.length() == 100

    context = RunContext()
    context.training = True
    context.max_train_count = 10
    trainer = rl_config.make_trainer(parameter, memory, env=env)
    c = mocker.Mock(spec=DummyTrainCallback)
    play_trainer_only(context, trainer, callbacks=[c])

    assert c.on_trainer_start.call_count == 1
    assert c.on_train_before.call_count == 10
    assert c.on_train_after.call_count == 10
    assert c.on_trainer_end.call_count == 1
