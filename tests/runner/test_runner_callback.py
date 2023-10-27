import pytest_mock

import srl
from srl.algorithms import ql, ql_agent57
from srl.runner.callback import Callback, TrainerCallback


def test_callback(mocker: pytest_mock.MockerFixture):
    env_config = srl.EnvConfig("Grid", frameskip=4)
    runner = srl.Runner(env_config, ql.Config())

    c = mocker.Mock(spec=Callback)
    runner.train(max_steps=10, callbacks=[c])
    assert runner.state.total_step == 10
    assert c.on_episodes_begin.call_count == 1
    assert c.on_episodes_end.call_count == 1
    assert c.on_episode_begin.call_count >= c.on_episode_end.call_count  # episode中に終了する可能性あり
    assert c.on_episode_end.call_count == runner.state.episode_count
    assert c.on_step_begin.call_count == 10
    assert c.on_step_begin.call_count == c.on_step_action_before.call_count
    assert c.on_step_begin.call_count == c.on_step_end.call_count
    assert c.on_skip_step.call_count > 30  # episode終了タイミングで変化する


def test_trainer_callback(mocker: pytest_mock.MockerFixture):
    env_config = srl.EnvConfig("Grid")
    rl_config = ql_agent57.Config()
    rl_config.memory.warmup_size = 100
    rl_config.batch_size = 1
    runner = srl.Runner(env_config, rl_config)
    runner.rollout(max_memory=100)

    c = mocker.Mock(spec=TrainerCallback)
    runner.train_only(max_train_count=10, callbacks=[c])

    assert c.on_trainer_start.call_count == 1
    assert c.on_trainer_train_end.call_count == 10
    assert c.on_trainer_end.call_count == 1
