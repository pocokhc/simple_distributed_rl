import ctypes
import multiprocessing as mp
import queue
from multiprocessing import sharedctypes
from typing import Optional, cast

import numpy as np
import pytest
import pytest_mock

import srl
from srl.algorithms import ql_agent57
from srl.base.context import RunContext
from srl.base.run.callback import RunCallback, TrainerCallback
from srl.base.run.core_play import RunStateActor
from srl.base.run.core_train_only import RunStateTrainer
from srl.runner.core_mp import _run_actor, _run_trainer
from srl.utils import common


class _AssertTrainCallbacks(RunCallback, TrainerCallback):
    def on_episodes_end(self, context: RunContext, state: RunStateActor) -> None:
        assert state.sync_actor > 1

    def on_trainer_end(self, context: RunContext, state: RunStateTrainer) -> None:
        assert state.sync_trainer > 1


class _DummyValue:
    def __init__(self, v) -> None:
        self.value = v


@pytest.mark.parametrize("interrupt_stop", [False, True])
@pytest.mark.timeout(5)  # pip install pytest_timeout
def test_actor(mocker: pytest_mock.MockerFixture, interrupt_stop: bool):
    common.logger_print()

    remote_queue = queue.Queue()
    remote_qsize = cast(sharedctypes.Synchronized, mp.Value(ctypes.c_int, 0))
    remote_board = _DummyValue(None)
    end_signal = _DummyValue(False)

    # --- create task
    c = mocker.Mock(spec=RunCallback)
    runner = srl.Runner("Grid", ql_agent57.Config())
    if not interrupt_stop:
        runner.context.max_episodes = 2
    runner.context.training = True
    runner.context.distributed = True
    runner.config.actor_parameter_sync_interval = 0

    task_config = runner.create_task_config(callbacks=[])
    task_config.callbacks.append(c)

    if interrupt_stop:

        class _c2(RunCallback):
            def __init__(self, end_signal):
                self.end_signal = end_signal

            def on_episode_end(self, context: RunContext, state: RunStateActor) -> None:
                self.end_signal.value = True

        task_config.callbacks.append(_c2(end_signal))

    # --- run
    _run_actor(
        task_config,
        remote_queue,
        remote_qsize,
        1000,
        remote_board,
        0,
        end_signal,
    )

    assert end_signal.value
    assert c.on_episodes_begin.call_count > 0
    assert c.on_episodes_end.call_count > 0
    assert remote_queue.qsize() > 0

    batch = remote_queue.get(timeout=1)
    print(batch)


@pytest.mark.parametrize("enable_prepare_sample_batch", [False, True])
@pytest.mark.parametrize("interrupt_stop", [False, True])
@pytest.mark.timeout(5)  # pip install pytest_timeout
def test_trainer(mocker: pytest_mock.MockerFixture, enable_prepare_sample_batch, interrupt_stop: bool):
    common.logger_print()

    remote_queue = queue.Queue()
    remote_qsize = cast(sharedctypes.Synchronized, mp.Value(ctypes.c_int, 0))
    remote_board = _DummyValue(None)
    end_signal = _DummyValue(False)

    # --- create task
    c = mocker.Mock(spec=TrainerCallback)
    rl_config = ql_agent57.Config()
    rl_config.memory.warmup_size = 10
    rl_config.batch_size = 1
    runner = srl.Runner("Grid", rl_config)
    if not interrupt_stop:
        runner.context.max_train_count = 10
    runner.context.timeout = 10
    runner.context.training = True
    runner.context.distributed = True
    runner.config.trainer_parameter_send_interval = 0
    runner.config.dist_enable_prepare_sample_batch = enable_prepare_sample_batch

    task_config = runner.create_task_config(callbacks=[])
    task_config.callbacks.append(c)

    if interrupt_stop:

        class _c2(TrainerCallback):
            def __init__(self, train_end_signal):
                self.train_end_signal = train_end_signal

            def on_train_after(self, context: RunContext, state: RunStateTrainer) -> None:
                assert state.trainer is not None
                if state.trainer.get_train_count() > 10:
                    self.train_end_signal.value = True

        task_config.callbacks.append(_c2(end_signal))

    # --- add queue
    for _ in range(100):
        remote_queue.put(
            (
                {
                    "states": ["1,3", "1,2"],
                    "actions": [3],
                    "probs": [0.25],
                    "ext_rewards": [-0.03999999910593033],
                    "int_rewards": [5.0],
                    "invalid_actions": [[], []],
                    "done": False,
                    "discount": 0.9999,
                },
                0,
            ),
        )

    # --- run
    _run_trainer(
        task_config,
        runner.make_parameter(),
        runner.make_memory(),
        remote_queue,
        remote_qsize,
        remote_board,
        end_signal,
    )

    assert end_signal.value
    assert c.on_trainer_start.call_count > 0
    assert c.on_trainer_end.call_count > 0


@pytest.mark.timeout(10)  # pip install pytest_timeout
def test_train():
    common.logger_print()
    rl_config = ql_agent57.Config(batch_size=2)
    rl_config.memory.warmup_size = 10
    runner = srl.Runner("Grid", rl_config)
    runner.set_progress_options(enable_eval=True)
    runner.train_mp(
        actor_num=2,
        max_train_count=20_000,
        callbacks=[_AssertTrainCallbacks()],
        trainer_parameter_send_interval=0,
        actor_parameter_sync_interval=0,
    )

    # eval
    rewards = runner.evaluate(max_episodes=100)
    rewards = np.mean(rewards)
    print(rewards)
    assert rewards > 0.5
