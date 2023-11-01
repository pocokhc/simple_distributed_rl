import pytest
import pytest_mock
import pytest_timeout  # noqa F401

import srl
from srl.algorithms import ql_agent57
from srl.runner.distribution import actor_run_forever, trainer_run_forever
from srl.runner.distribution.callback import ActorServerCallback, DistributionCallback, TrainerServerCallback
from srl.runner.distribution.client import run as dist_run
from srl.runner.distribution.manager import DistributedManager, ServerParameters
from srl.runner.runner import Runner
from srl.utils import common
from tests.runner.distribution.server_mock import create_mock


@pytest.mark.timeout(10)  # pip install pytest_timeout
def test_client(mocker: pytest_mock.MockerFixture):
    create_mock(mocker)

    common.logger_print()

    runner = srl.Runner("Grid", ql_agent57.Config())

    class _call(DistributionCallback):
        def __init__(self) -> None:
            super().__init__()
            self.count = 0

        def on_polling(self, runner: Runner, manager: DistributedManager, task_id: str) -> bool | None:
            self.manager = manager
            self.task_id = task_id
            manager.task_end(task_id)
            return False

    c = mocker.Mock(spec=DistributionCallback)
    c2 = _call()
    params = ServerParameters(keepalive_interval=0)
    dist_run(runner, params, callbacks=[c, c2])

    assert c.on_start.call_count == 1
    assert c.on_polling.call_count > 0
    assert c.on_end.call_count == 1

    assert c2.manager.task_get_config(c2.task_id) is not None


@pytest.mark.parametrize("server", ["redis", "pika"])
@pytest.mark.parametrize("enable_actor_thread", [False, True])
@pytest.mark.timeout(30)  # pip install pytest_timeout
def test_server_actor(mocker: pytest_mock.MockerFixture, server, enable_actor_thread):
    create_mock(mocker)
    if server == "redis":
        rabbitmq_host = ""
    else:
        rabbitmq_host = "test"

    common.logger_print()

    # --- create task
    runner = srl.Runner("Grid", ql_agent57.Config())
    runner.context.timeout = 1
    runner.context.training = True
    runner.config.dist_enable_actor_thread = enable_actor_thread
    runner.config.actor_parameter_sync_interval = 0
    manager = DistributedManager(ServerParameters(rabbitmq_host=rabbitmq_host, keepalive_interval=0))
    manager.set_user("client")
    manager.task_add(1, runner.create_task_config(), runner.make_parameter().backup())

    # --- run
    c2 = mocker.Mock(spec=ActorServerCallback)
    params = ServerParameters(rabbitmq_host=rabbitmq_host, keepalive_interval=0)
    actor_run_forever(params, callbacks=[c2], run_once=True)

    assert c2.on_polling.call_count > 0

    remote_memory = manager.create_memory_connector()
    assert remote_memory.memory_size(manager.uid) > 0

    batch = remote_memory.memory_recv(manager.uid)
    print(batch)


@pytest.mark.parametrize("server", ["redis", "pika"])
@pytest.mark.parametrize("dist_option", [[False, False], [True, False], [True, True]])
@pytest.mark.timeout(10)  # pip install pytest_timeout
def test_server_trainer(mocker: pytest_mock.MockerFixture, server, dist_option):
    create_mock(mocker)
    if server == "redis":
        rabbitmq_host = ""
    else:
        rabbitmq_host = "test"

    common.logger_print()

    # --- create task
    rl_config = ql_agent57.Config()
    rl_config.memory.warmup_size = 10
    rl_config.batch_size = 1
    runner = srl.Runner("Grid", rl_config)
    runner.context.max_train_count = 5
    runner.context.training = True
    runner.config.dist_enable_trainer_thread = dist_option[0]
    runner.config.dist_enable_prepare_sample_batch = dist_option[1]
    runner.config.trainer_parameter_send_interval = 0
    manager = DistributedManager(ServerParameters(rabbitmq_host=rabbitmq_host, keepalive_interval=0))
    manager.set_user("client")
    manager.task_add(1, runner.create_task_config(), runner.make_parameter().backup())

    # add batch
    remote_memory = manager.create_memory_connector()
    for _ in range(100):
        remote_memory.memory_add(
            manager.uid,
            (
                {
                    "states": ["1,3", "1,3"],
                    "actions": [0],
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
    assert manager.task_get_trainer(manager.uid, "train") == ""
    c2 = mocker.Mock(spec=TrainerServerCallback)
    params = ServerParameters(rabbitmq_host=rabbitmq_host, keepalive_interval=0)
    trainer_run_forever(params, callbacks=[c2], run_once=True)
    assert c2.on_polling.call_count > 0
    assert int(manager.task_get_trainer(manager.uid, "train")) > 0
