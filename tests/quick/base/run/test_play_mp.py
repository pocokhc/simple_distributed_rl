import ctypes
import multiprocessing as mp
import queue
from multiprocessing import sharedctypes
from typing import cast

import pytest
import pytest_mock

import srl
from srl.algorithms import ql_agent57
from srl.base.context import RunContext
from srl.base.run.callback import RunCallback
from srl.base.run.core_play import RunStateActor
from srl.base.run.core_train_only import RunStateTrainer
from srl.base.run.play_mp import MpConfig, _run_actor, _run_trainer


class _DummyValue:
    def __init__(self, v) -> None:
        self.value = v

    def set(self, d):
        self.value = d


@pytest.mark.parametrize("interrupt_stop", [False, True])
def test_actor(mocker: pytest_mock.MockerFixture, interrupt_stop: bool):
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

    mp_cfg = MpConfig(
        runner.context,
        [c],
        actor_parameter_sync_interval=0,
    )

    if interrupt_stop:

        class _c2(RunCallback):
            def __init__(self, end_signal):
                self.end_signal = end_signal

            def on_episode_end(self, context: RunContext, state: RunStateActor) -> None:
                self.end_signal.value = True

        mp_cfg.callbacks.append(_c2(end_signal))

    # --- run
    _run_actor(
        mp_cfg,
        remote_queue,
        remote_qsize,
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


@pytest.mark.parametrize("interrupt_stop", [False, True])
def test_trainer(mocker: pytest_mock.MockerFixture, interrupt_stop: bool):
    remote_queue = queue.Queue()
    remote_qsize = cast(sharedctypes.Synchronized, mp.Value(ctypes.c_int, 0))
    remote_board = _DummyValue(None)
    end_signal = _DummyValue(False)
    last_mem_queue = queue.Queue()  # mp.Queue()を使うとhungする

    # --- create task
    c = mocker.Mock(spec=RunCallback)
    rl_config = ql_agent57.Config()
    rl_config.memory.warmup_size = 10
    rl_config.batch_size = 1
    runner = srl.Runner("Grid", rl_config)
    if not interrupt_stop:
        runner.context.max_train_count = 10
    runner.context.timeout = 10
    runner.context.training = True
    runner.context.distributed = True
    mp_cfg = MpConfig(
        runner.context,
        [c],
        trainer_parameter_send_interval=0,
    )

    if interrupt_stop:

        class _c2(RunCallback):
            def __init__(self, train_end_signal):
                self.train_end_signal = train_end_signal

            def on_train_after(self, context: RunContext, state: RunStateTrainer) -> None:
                assert state.trainer is not None
                if state.trainer.get_train_count() > 10:
                    self.train_end_signal.value = True

        mp_cfg.callbacks.append(_c2(end_signal))

    # --- add queue
    serialize_func = runner.make_memory().get_worker_funcs()["add"][1]
    for _ in range(100):
        batch = {
            "states": ["1,3", "1,2"],
            "actions": [3],
            "probs": [0.25],
            "ext_rewards": [-0.03999999910593033],
            "int_rewards": [5.0],
            "invalid_actions": [[], []],
            "done": False,
            "discount": 0.9999,
        }
        remote_queue.put(("add", serialize_func(batch, -1)))

    # --- run
    _run_trainer(
        mp_cfg,
        remote_queue,
        remote_qsize,
        remote_board,
        end_signal,
        None,
        last_mem_queue,  # type: ignore
    )

    assert end_signal.value
    assert c.on_trainer_start.call_count > 0
    assert c.on_trainer_end.call_count > 0
