import ctypes
import multiprocessing as mp
import queue
from typing import cast

import numpy as np
import pytest
import pytest_mock

import srl
from srl.algorithms import ql_agent57
from srl.runner.callback import Callback, TrainerCallback
from srl.runner.core_mp import _Board, _run_actor, _run_trainer
from srl.runner.runner import Runner
from srl.utils import common


class _AssertTrainCallbacks(Callback, TrainerCallback):
    def on_episodes_end(self, runner: srl.Runner) -> None:
        assert runner.state.sync_actor > 1

    def on_trainer_end(self, runner: srl.Runner) -> None:
        assert runner.state.sync_trainer > 1


@pytest.mark.parametrize("interrupt_stop", [False, True])
@pytest.mark.timeout(5)  # pip install pytest_timeout
def test_actor(mocker: pytest_mock.MockerFixture, interrupt_stop: bool):
    common.logger_print()

    memory_queue = queue.Queue()
    remote_board = mocker.Mock(spec=_Board)
    remote_board.read.return_value = None
    train_end_signal = cast(ctypes.c_bool, mp.Value(ctypes.c_bool, False))

    # --- create task
    c = mocker.Mock(spec=Callback)
    runner = srl.Runner("Grid", ql_agent57.Config())
    if not interrupt_stop:
        runner.context.max_episodes = 2
    runner.context.training = True
    runner.context.distributed = True
    runner.config.actor_parameter_sync_interval = 0
    runner.context.callbacks = [c]

    if interrupt_stop:

        class _c2(Callback):
            def __init__(self, train_end_signal: ctypes.c_bool):
                self.train_end_signal = train_end_signal

            def on_episode_end(self, runner: Runner) -> None:
                self.train_end_signal.value = True

        runner.context.callbacks.append(_c2(train_end_signal))

    # --- run
    _run_actor(runner.create_task_config(), memory_queue, remote_board, 0, train_end_signal)

    assert train_end_signal.value
    assert c.on_episodes_begin.call_count > 0
    assert c.on_episodes_end.call_count > 0
    assert remote_board.read.call_count > 0
    assert memory_queue.qsize() > 0

    batch = memory_queue.get(timeout=1)
    print(batch)


@pytest.mark.parametrize("enable_prepare_sample_batch", [False, True])
@pytest.mark.parametrize("interrupt_stop", [False, True])
@pytest.mark.timeout(5)  # pip install pytest_timeout
def test_trainer(mocker: pytest_mock.MockerFixture, enable_prepare_sample_batch, interrupt_stop: bool):
    common.logger_print()

    memory_queue = queue.Queue()
    remote_board = mocker.Mock(spec=_Board)
    train_end_signal = cast(ctypes.c_bool, mp.Value(ctypes.c_bool, False))

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
    runner.context.callbacks = [c]

    if interrupt_stop:

        class _c2(TrainerCallback):
            def __init__(self, train_end_signal: ctypes.c_bool):
                self.train_end_signal = train_end_signal

            def on_trainer_loop(self, runner: Runner) -> None:
                assert runner.state.trainer is not None
                if runner.state.trainer.get_train_count() > 10:
                    self.train_end_signal.value = True

        runner.context.callbacks.append(_c2(train_end_signal))

    # --- add queue
    for _ in range(100):
        memory_queue.put(
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
        runner.create_task_config(),
        runner.make_parameter(),
        runner.make_memory(),
        memory_queue,
        remote_board,
        train_end_signal,
    )

    assert train_end_signal.value
    assert c.on_trainer_start.call_count > 0
    assert c.on_trainer_end.call_count > 0
    assert remote_board.write.call_count > 0


@pytest.mark.timeout(5)  # pip install pytest_timeout
def test_train():
    rl_config = ql_agent57.Config(batch_size=2)
    rl_config.memory.warmup_size = 10
    runner = srl.Runner("Grid", rl_config)
    runner.train_mp(
        actor_num=2,
        max_train_count=10_000,
        enable_eval=True,
        callbacks=[_AssertTrainCallbacks()],
        trainer_parameter_send_interval=0,
        actor_parameter_sync_interval=0,
    )

    # eval
    rewards = runner.evaluate(max_episodes=100)
    rewards = np.mean(rewards)
    print(rewards)
    assert rewards > 0.5
