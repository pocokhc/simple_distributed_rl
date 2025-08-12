import multiprocessing as mp
import socket
import time

import numpy as np
import pytest
import pytest_timeout  # noqa F401

import srl
from srl.algorithms import ql_agent57
from srl.base.context import RunContext
from srl.base.run.callback import RunCallback
from srl.base.run.core_play import RunStateActor
from srl.base.run.core_train_only import RunStateTrainer
from srl.runner.distribution.connector_configs import MQTTParameters, RabbitMQParameters, RedisParameters
from srl.runner.distribution.server_actor import run_forever as actor_run_forever
from srl.runner.distribution.server_manager import TaskManager
from srl.runner.distribution.server_trainer import run_forever as trainer_run_forever


def is_port_open(host, port):
    try:
        socket.create_connection((host, port), timeout=1)
        return True
    except (ConnectionRefusedError, TimeoutError):
        return False


def _run_actor(memory_params):
    actor_run_forever(
        RedisParameters(host="localhost"),
        memory_params,
        keepalive_interval=1,
    )


def _run_trainer(memory_params):
    trainer_run_forever(
        RedisParameters(host="localhost"),
        memory_params,
        keepalive_interval=1,
    )


class _AssertTrainCallbacks(RunCallback):
    def on_episodes_end(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        assert state.sync_actor > 0

    def on_trainer_end(self, context: RunContext, state: RunStateTrainer, **kwargs) -> None:
        assert state.sync_trainer > 0


@pytest.mark.parametrize("is_wait", [False, True])
@pytest.mark.parametrize(
    "memory_params",
    [
        None,
        RedisParameters(host="localhost"),
        RabbitMQParameters(host="localhost", ssl=False),
        MQTTParameters(host="localhost"),
    ],
)
@pytest.mark.timeout(60)  # pip install pytest_timeout
def test_train(is_wait, memory_params):
    pytest.importorskip("redis")

    # 起動しないテスト方法が不明...
    # サーバが起動している事

    assert is_port_open("127.0.0.1", 6379), "Redis is not running."

    th_actor = mp.Process(target=_run_actor, args=(memory_params,))
    th_trainer = mp.Process(target=_run_trainer, args=(memory_params,))

    th_actor.start()
    th_trainer.start()
    try:
        time.sleep(1)

        runner = srl.Runner("Grid", ql_agent57.Config(batch_size=2))
        if is_wait:
            runner.train_distribution(
                RedisParameters(host="localhost"),
                trainer_parameter_send_interval=0,
                actor_parameter_sync_interval=0,
                max_train_count=200_000,
                # callbacks=[_AssertTrainCallbacks()],  # 分散先ではlocalは読み込めない
            )
        else:
            runner.train_distribution_start(
                RedisParameters(host="localhost"),
                trainer_parameter_send_interval=0,
                actor_parameter_sync_interval=0,
                max_train_count=200_000,
                # callbacks=[_AssertTrainCallbacks()],  # 分散先ではlocalは読み込めない
            )
            task_manager = TaskManager(RedisParameters(host="localhost"))
            task_manager.train_wait()
            runner = task_manager.create_runner()
            assert runner is not None

        # eval
        rewards = runner.evaluate(max_episodes=100)
        rewards = np.mean(rewards)
        print(rewards)
        assert rewards > 0.4

    finally:
        th_actor.terminate()
        th_trainer.terminate()
        th_actor.join()
        th_trainer.join()
