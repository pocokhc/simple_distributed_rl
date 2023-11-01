import multiprocessing as mp
import socket
import time

import numpy as np
import pytest
import pytest_timeout  # noqa F401

import srl
from srl.algorithms import ql_agent57
from srl.runner.callback import Callback, TrainerCallback
from srl.runner.distribution.manager import ServerParameters
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
    actor_run_forever(ServerParameters(redis_host="localhost", rabbitmq_host="localhost"))


def _run_trainer():
    common.logger_print()
    trainer_run_forever(ServerParameters(redis_host="localhost", rabbitmq_host="localhost"))


class _AssertTrainCallbacks(Callback, TrainerCallback):
    def on_episodes_end(self, runner: srl.Runner) -> None:
        assert runner.state.sync_actor > 0

    def on_trainer_end(self, runner: srl.Runner) -> None:
        assert runner.state.sync_trainer > 0


@pytest.mark.timeout(60)  # pip install pytest_timeout
def test_train():
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
        ServerParameters(redis_host="localhost", keepalive_interval=0),
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
