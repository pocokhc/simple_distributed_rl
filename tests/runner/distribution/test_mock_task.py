import pytest
import pytest_mock

import srl
from srl.algorithms import ql
from srl.base.run.context import RunContext
from srl.runner.distribution.connectors.parameters import (
    GCPParameters,
    MQTTParameters,
    RabbitMQParameters,
    RedisParameters,
)
from srl.runner.distribution.task_manager import TaskManager
from srl.runner.runner import RunnerConfig, TaskConfig
from tests.runner.distribution.memory_test_functions import memory_connector_test
from tests.runner.distribution.server_mock import (
    create_gcp_mock,
    create_paho_mock,
    create_pika_mock,
    create_redis_mock,
)


def test_memory_redis(mocker: pytest_mock.MockerFixture):
    pytest.importorskip("redis")
    create_redis_mock(mocker)
    memory_connector_test(RedisParameters(host="test"))


def test_memory_pika(mocker: pytest_mock.MockerFixture):
    pytest.importorskip("pika")
    create_pika_mock(mocker)
    memory_connector_test(RabbitMQParameters(host="test"))


def test_memory_mqtt(mocker: pytest_mock.MockerFixture):
    pytest.importorskip("paho")
    create_paho_mock(mocker)
    memory_connector_test(MQTTParameters(host="test"))


def test_memory_gcp(mocker: pytest_mock.MockerFixture):
    pytest.importorskip("google")
    create_gcp_mock(mocker)
    memory_connector_test(GCPParameters(project_id="test"))


def test_parameter(mocker: pytest_mock.MockerFixture):
    pytest.importorskip("redis")
    create_redis_mock(mocker)

    params = RedisParameters(host="test")
    connector = params.create_connector()

    connector.parameter_update("a")
    assert connector.parameter_read() == "a"


def test_no_task(mocker: pytest_mock.MockerFixture):
    pytest.importorskip("redis")
    create_redis_mock(mocker)

    t = TaskManager(RedisParameters(host="test"), "client")

    t.finished()
    assert t.get_status() == "END"
    assert t.get_actor_num() == -1
    assert t.get_trainer("a") == ""
    t.set_trainer("a", "a")
    assert t.get_trainer("a") == "a"
    assert t.get_actor(1, "a") == ""
    t.set_actor(1, "a", "a")
    assert t.get_actor(1, "a") == "a"
    t.add_log("a")


def test_task(mocker: pytest_mock.MockerFixture):
    pytest.importorskip("redis")
    create_redis_mock(mocker)

    client = TaskManager(RedisParameters(host="test"), "client")
    task_config = TaskConfig(RunnerConfig(), RunContext(srl.EnvConfig("Grid"), ql.Config()), [])
    task_config.context.actor_num = 2
    task_config.context.max_train_count = 10
    client.create(task_config, "parameter")
    assert client.get_status() == "ACTIVE"
    assert client.get_actor_num() == 2
    assert client.get_trainer("id") == ""
    assert client.get_actor(0, "id") == ""
    assert client.get_actor(1, "id") == ""
    assert client.get_actor(-1, "id") == ""

    # --- assign trainer
    trainer = TaskManager(RedisParameters(host="test"), "trainer")
    from srl.runner.distribution.server_trainer import _task_assign as trainer_task_assign

    assert trainer_task_assign(trainer)
    assert trainer.get_config() is not None
    assert client.get_trainer("id") == trainer.params.uid

    assert not trainer_task_assign(trainer)

    # --- assign actor1
    actor1 = TaskManager(RedisParameters(host="test"), "actor")
    from srl.runner.distribution.server_actor import _task_assign as actor_task_assign

    # queueのセットアップ前はアサインしない
    is_assigned, actor_id1 = actor_task_assign(actor1)
    assert not is_assigned

    # queueをセットアップ
    assert not trainer.is_setup_memory()
    trainer.setup_memory(RedisParameters(host="test").create_memory_receiver(), is_purge=True)
    assert trainer.is_setup_memory()

    is_assigned, actor_id1 = actor_task_assign(actor1)
    assert is_assigned
    assert actor_id1 == 0
    assert actor1.get_config() is not None
    assert client.get_actor(actor_id1, "id") == actor1.params.uid

    is_assigned, actor_id1 = actor_task_assign(actor1)
    assert is_assigned
    assert actor_id1 == 0

    # assign actor2
    actor2 = TaskManager(RedisParameters(host="test"), "actor")
    is_assigned, actor_id2 = actor_task_assign(actor2)
    assert is_assigned
    assert actor_id2 == 1
    assert actor2.get_config() is not None
    assert client.get_actor(actor_id2, "id") == actor2.params.uid

    is_assigned, actor_id2 = actor_task_assign(actor2)
    assert is_assigned
    assert actor_id2 == 1

    # assign actor3
    actor3 = TaskManager(RedisParameters(host="test"), "actor")
    is_assigned, _ = actor_task_assign(actor3)
    assert not is_assigned

    # finish
    assert not client.is_finished()
    client.set_train_count(0, 100)
    assert client.is_finished()
    assert client.get_status() == "END"
