import ctypes
import multiprocessing as mp
from multiprocessing import sharedctypes
from typing import cast

import numpy as np
import pytest
import pytest_mock

import srl
from srl.algorithms import ql_agent57
from srl.base.context import RunContext
from srl.base.run.callback import RunCallback
from srl.base.run.core_play import RunStateActor
from srl.base.run.core_train_only import RunStateTrainer
from srl.base.run.play_mp_memory import MpConfig, _run_actor


class _AssertTrainCallbacks(RunCallback):
    def on_episodes_end(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        assert state.sync_actor > 1

    def on_trainer_end(self, context: RunContext, state: RunStateTrainer, **kwargs) -> None:
        assert state.sync_trainer > 1


class _DummyValue:
    def __init__(self, v) -> None:
        self.value = v

    def set(self, d):
        self.value = d


@pytest.mark.parametrize("interrupt_stop", [False, True])
@pytest.mark.timeout(5)  # pip install pytest_timeout
def test_actor(mocker: pytest_mock.MockerFixture, interrupt_stop: bool):
    remote_queue = mp.Queue()
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


@pytest.mark.timeout(30)  # pip install pytest_timeout
def test_train():
    rl_config = ql_agent57.Config(batch_size=1)
    rl_config.memory.warmup_size = 100
    runner = srl.Runner("Grid", rl_config)
    runner.set_progress(enable_eval=True)
    runner.train_mp(
        actor_num=1,
        max_train_count=30_000,
        callbacks=[_AssertTrainCallbacks()],
        trainer_parameter_send_interval=0,
        actor_parameter_sync_interval=0,
        enable_mp_memory=True,
    )

    # eval
    rewards = runner.evaluate(max_episodes=100)
    rewards = np.mean(rewards)
    print(rewards)
    assert rewards > 0.5
