import multiprocessing as mp
import socket
import time

import numpy as np
import pytest
import pytest_timeout  # noqa F401

import srl
from srl.algorithms import ql_agent57
from srl.base.run.callback import RunCallback, TrainerCallback
from srl.base.run.context import RunContext
from srl.base.run.core import RunState
from srl.runner.distribution.connectors.parameters import RabbitMQParameters, RedisParameters
from srl.runner.distribution.server_actor import run_forever as actor_run_forever
from srl.runner.distribution.server_trainer import run_forever as trainer_run_forever
from srl.utils import common


def is_port_open(host, port):
    try:
        socket.create_connection((host, port), timeout=1)
        return True
    except (ConnectionRefusedError, TimeoutError):
        return False


def _run_actor():
    common.logger_print()
    actor_run_forever(
        RedisParameters(host="localhost"),
        RabbitMQParameters(host="localhost", ssl=False),
        keepalive_interval=0,
        run_once=True,
    )


def _run_trainer():
    common.logger_print()
    trainer_run_forever(
        RedisParameters(host="localhost"),
        RabbitMQParameters(host="localhost", ssl=False),
        keepalive_interval=0,
        run_once=True,
    )


class _AssertTrainCallbacks(RunCallback, TrainerCallback):
    def on_episodes_end(self, context: RunContext, state: RunState) -> None:
        assert state.sync_actor > 0

    def on_trainer_end(self, context: RunContext, state: RunState) -> None:
        assert state.sync_trainer > 0


@pytest.mark.timeout(60)  # pip install pytest_timeout
def test_train():
    pytest.importorskip("redis")

    # 起動しないテスト方法が不明...
    # サーバが起動している事
    common.logger_print()

    assert is_port_open("127.0.0.1", 5672), "RabbitMQ is not running."
    assert is_port_open("127.0.0.1", 6379), "Redis is not running."

    th_actor = mp.Process(target=_run_actor)
    th_trainer = mp.Process(target=_run_trainer)

    th_actor.start()
    th_trainer.start()

    time.sleep(1)

    runner = srl.Runner("Grid", ql_agent57.Config(batch_size=2))
    runner.train_distribution(
        RedisParameters(host="localhost"),
        trainer_parameter_send_interval=0,
        actor_parameter_sync_interval=0,
        max_train_count=100_000,
        callbacks=[_AssertTrainCallbacks()],
    )

    # eval
    rewards = runner.evaluate(max_episodes=100)
    rewards = np.mean(rewards)
    print(rewards)
    assert rewards > 0.4
