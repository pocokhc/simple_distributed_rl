import ctypes
import multiprocessing as mp
import socket
import time
from typing import cast

import numpy as np

import srl
from srl.algorithms import ql_agent57
from srl.runner.callback import Callback, TrainerCallback
from srl.runner.distribution.server_actor import ActorServerCallback
from srl.runner.distribution.server_actor import run_forever as actor_run_forever
from srl.runner.distribution.server_trainer import TrainerServerCallback
from srl.runner.distribution.server_trainer import run_forever as trainer_run_forever
from srl.utils import common


def is_port_open(host, port):
    try:
        socket.create_connection((host, port), timeout=1)
        return True
    except (ConnectionRefusedError, TimeoutError):
        return False


class _ActorServerCallback(ActorServerCallback):
    def __init__(self, end_signal: ctypes.c_bool):
        self.end_signal = end_signal
        self.t0 = time.time()

    def on_polling(self) -> bool:
        if time.time() - self.t0 > 10:
            return True
        return self.end_signal.value


class _TrainerServerCallback(TrainerServerCallback):
    def __init__(self, end_signal: ctypes.c_bool):
        self.end_signal = end_signal
        self.t0 = time.time()

    def on_polling(self) -> bool:
        if time.time() - self.t0 > 10:
            return True
        return self.end_signal.value


def _run_actor(end_signal):
    actor_run_forever("127.0.0.1", callbacks=[_ActorServerCallback(end_signal)])


def _run_trainer(end_signal):
    trainer_run_forever("127.0.0.1", callbacks=[_TrainerServerCallback(end_signal)])


class _AssertTrainCallbacks(Callback, TrainerCallback):
    def on_episodes_end(self, runner: srl.Runner) -> None:
        assert runner.state.sync_actor > 0

    def on_trainer_end(self, runner: srl.Runner) -> None:
        # assert runner.state.sync_trainer > 0  # TODO
        pass


def test_train():
    # 起動しないテスト方法が不明...
    # サーバが起動している事
    common.logger_print()

    assert is_port_open("127.0.0.1", 6379), "Redis is not running."

    end_signal = cast(ctypes.c_bool, mp.Value(ctypes.c_bool, False))
    th_actor = mp.Process(target=_run_actor, args=(end_signal,))
    th_trainer = mp.Process(target=_run_trainer, args=(end_signal,))

    th_actor.start()
    th_trainer.start()

    time.sleep(1)

    runner = srl.Runner("Grid", ql_agent57.Config(batch_size=2))
    runner.train_distribution(
        "127.0.0.1",
        max_train_count=100_000,
        timeout=10,  # for safety
        callbacks=[_AssertTrainCallbacks()],
    )
    end_signal.value = True

    # eval
    rewards = runner.evaluate(max_episodes=100)
    rewards = np.mean(rewards)
    print(rewards)
    assert rewards > 0.4
