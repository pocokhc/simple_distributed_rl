from typing import Optional

import pytest
import pytest_mock
import pytest_timeout  # noqa F401

import srl
from srl.algorithms import ql, ql_agent57
from srl.base.context import RunContext
from srl.base.run.callback import RunCallback
from srl.base.run.core_play import RunStateActor, play
from srl.base.run.core_train_only import RunStateTrainer, play_trainer_only


class DummyCallback(RunCallback):
    def on_episode_begin(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        pass  # do nothing

    def on_episode_end(self, context: RunContext, state: RunStateActor) -> None:
        pass  # do nothing

    def on_step_action_before(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        pass  # do nothing

    def on_step_action_after(self, context: RunContext, state: RunStateActor) -> None:
        pass  # do nothing

    def on_step_begin(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        pass  # do nothing

    def on_step_end(self, context: RunContext, state: RunStateActor) -> Optional[bool]:
        return False

    def on_skip_step(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        pass  # do nothing

    def on_train_before(self, context: RunContext, state: RunStateTrainer, **kwargs):
        pass

    def on_train_after(self, context: RunContext, state: RunStateTrainer, **kwargs) -> Optional[bool]:
        return False


def test_callback(mocker: pytest_mock.MockerFixture):
    c = mocker.Mock(spec=DummyCallback)

    context = RunContext(srl.EnvConfig("Grid", frameskip=4), ql.Config())
    context.training = True
    context.max_episodes = 1
    context.callbacks = [c]
    env = context.env_config.make()
    context.rl_config.setup(env)
    state = play(context, env, context.rl_config.make_worker(env, context.rl_config.make_parameter(), context.rl_config.make_memory()))

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


@pytest.mark.timeout(2)  # pip install pytest_timeout
def test_trainer_callback(mocker: pytest_mock.MockerFixture):
    env_config = srl.EnvConfig("Grid")
    rl_config = ql_agent57.Config()
    rl_config.memory.warmup_size = 100
    rl_config.batch_size = 1

    context = RunContext(env_config, rl_config)
    context.training = True
    context.max_memory = 100
    context.disable_trainer = True
    env = env_config.make()
    rl_config.setup(env)
    state = play(context, env, rl_config.make_worker(env, rl_config.make_parameter(), rl_config.make_memory()))
    assert state.memory.length() >= 100

    context.training = True
    context.max_train_count = 10
    c = mocker.Mock(spec=DummyCallback)
    context.callbacks = [c]
    play_trainer_only(context, rl_config.make_trainer(state.parameter, state.memory))

    assert c.on_trainer_start.call_count == 1
    assert c.on_train_before.call_count == 10
    assert c.on_train_after.call_count == 10
    assert c.on_trainer_end.call_count == 1
