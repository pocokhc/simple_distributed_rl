import pytest
import pytest_mock
import pytest_timeout  # noqa F401

import srl
from srl.algorithms import ql, ql_agent57
from srl.base.context import RunContext
from srl.base.run.callback import RunCallback, TrainerCallback
from srl.base.run.core_play import play
from srl.base.run.core_train_only import play_trainer_only


def test_callback(mocker: pytest_mock.MockerFixture):
    env_config = srl.EnvConfig("Grid", frameskip=4)
    rl_config = ql.Config()

    env = srl.make_env(env_config)
    parameter = srl.make_parameter(rl_config, env)
    memory = srl.make_memory(rl_config, env)
    trainer = srl.make_trainer(rl_config, parameter, memory)
    c = mocker.Mock(spec=RunCallback)

    context = RunContext()
    context.training = True
    context.max_episodes = 1
    workers, main_worker_idx = context.make_workers(env, parameter, memory, rl_config)
    state = play(context, env, workers, main_worker_idx, trainer=trainer, callbacks=[c])

    assert state.total_step >= 1
    assert state.episode_count == 1
    assert c.on_episodes_begin.call_count == 1
    assert c.on_episodes_end.call_count == 1
    assert c.on_episode_begin.call_count >= c.on_episode_end.call_count  # episode中に終了する可能性あり
    assert c.on_episode_end.call_count == 1
    assert c.on_step_begin.call_count >= 1
    assert c.on_step_begin.call_count == c.on_step_action_before.call_count
    assert c.on_step_begin.call_count == c.on_step_end.call_count
    assert c.on_skip_step.call_count >= 1  # episode終了タイミングで変化する


@pytest.mark.timeout(2)  # pip install pytest_timeout
def test_trainer_callback(mocker: pytest_mock.MockerFixture):
    env_config = srl.EnvConfig("Grid")
    rl_config = ql_agent57.Config()
    rl_config.memory.warmup_size = 100
    rl_config.batch_size = 1

    env = srl.make_env(env_config)
    parameter = srl.make_parameter(rl_config, env)
    memory = srl.make_memory(rl_config, env)

    context = RunContext()
    context.training = True
    context.max_memory = 100
    context.disable_trainer = True
    workers, main_worker_idx = context.make_workers(env, parameter, memory, rl_config)
    play(context, env, workers, main_worker_idx)
    assert memory.length() == 100

    context = RunContext()
    context.training = True
    context.max_train_count = 10
    trainer = srl.make_trainer(rl_config, parameter, memory, env=env)
    c = mocker.Mock(spec=TrainerCallback)
    play_trainer_only(context, trainer, callbacks=[c])

    assert c.on_trainer_start.call_count == 1
    assert c.on_trainer_loop.call_count == 10
    assert c.on_trainer_end.call_count == 1
