from typing import Literal

import pytest
import pytest_mock

from srl.runner.distribution.connectors.parameters import GCPParameters, RabbitMQParameters, RedisParameters
from srl.runner.distribution.manager import DistributedManager
from tests.runner.distribution.server_mock import create_gcp_mock, create_pika_mock, create_redis_mock
from tests.runner.distribution.test_connectors import memory_connector_test


def test_memory_redis(mocker: pytest_mock.MockerFixture):
    create_redis_mock(mocker)
    memory_connector_test(RedisParameters(host="test").create_memory_connector())


def test_memory_pika(mocker: pytest_mock.MockerFixture):
    create_pika_mock(mocker)
    memory_connector_test(RabbitMQParameters(host="test").create_memory_connector())


def test_memory_gcp(mocker: pytest_mock.MockerFixture):
    create_gcp_mock(mocker)
    memory_connector_test(GCPParameters(project_id="test").create_memory_connector())


def test_parameter(mocker: pytest_mock.MockerFixture):
    create_redis_mock(mocker)
    manager = DistributedManager(RedisParameters(host="test"), None)
    manager.set_user("trainer")

    assert manager.parameter_read("0") is None

    # 自分のidが登録されている場合のみ更新
    with pytest.raises(AssertionError):
        manager.parameter_update("0", "a")


@pytest.mark.parametrize("server", ["", "redis", "pika", "gcp"])
def test_no_task(mocker: pytest_mock.MockerFixture, server: Literal["redis", "pika", "gcp"]):
    create_redis_mock(mocker)
    if server == "":
        params = None
    elif server == "redis":
        params = RedisParameters(host="test")
    elif server == "pika":
        create_pika_mock(mocker)
        params = RabbitMQParameters(host="test")
    elif server == "gcp":
        create_gcp_mock(mocker)
        params = GCPParameters(project_id="test")
    else:
        raise
    manager = DistributedManager(RedisParameters(host="test"), params)

    manager.task_end("0")
    assert manager.task_get_status("0") == ""
    assert manager.task_get_actor_num("0") == -1
    assert manager.task_get_trainer("0", "a") == ""
    manager.task_set_trainer("0", "a", "a")
    assert manager.task_get_actor("0", 1, "a") == ""
    manager.task_set_actor("0", 1, "a", "a")
    manager.task_assign_by_my_id()
    manager.task_log("0", "")
    manager.keepalive("0")


@pytest.mark.parametrize("server", ["", "redis", "pika", "gcp"])
def test_task(mocker: pytest_mock.MockerFixture, server: Literal["redis", "pika", "gcp"]):
    create_redis_mock(mocker)
    if server == "":
        params = None
    elif server == "redis":
        params = RedisParameters(host="test")
    elif server == "pika":
        create_pika_mock(mocker)
        params = RabbitMQParameters(host="test")
    elif server == "gcp":
        create_gcp_mock(mocker)
        params = GCPParameters(project_id="test")
    else:
        raise

    m_client = DistributedManager(RedisParameters(host="test"), params)
    m_client.set_user("client")
    task_id = m_client.task_add(2, "config", "parameter")
    assert m_client.task_get_status(task_id) == "WAIT"
    assert m_client.task_get_actor_num(task_id) == 2
    assert m_client.task_get_trainer(task_id, "id") == "NO_ASSIGN"
    assert m_client.task_get_actor(task_id, 0, "id") == "NO_ASSIGN"
    assert m_client.task_get_actor(task_id, 1, "id") == "NO_ASSIGN"
    assert m_client.task_get_actor(task_id, -1, "id") == ""

    # assign trainer
    m_trainer = DistributedManager(RedisParameters(host="test"), params)
    m_trainer.set_user("trainer")
    task_id_trainer, actor_id = m_trainer.task_assign_by_my_id()
    assert task_id_trainer == task_id
    assert m_trainer.task_get_config(task_id) == "config"
    assert actor_id == 0
    assert m_trainer.task_get_trainer(task_id, "id") == m_trainer.uid
    assert m_trainer.parameter_read(task_id) == "parameter"
    m_trainer.parameter_update(task_id, "parameter2")

    _t, _ = m_trainer.task_assign_by_my_id()
    assert _t == ""

    # assign actor1
    m_actor1 = DistributedManager(RedisParameters(host="test"), params)
    m_actor1.set_user("actor")
    task_id_actor1, actor_id1 = m_actor1.task_assign_by_my_id()
    assert task_id_actor1 == task_id
    assert m_actor1.task_get_config(task_id) == "config"
    assert actor_id1 == 0
    assert m_actor1.task_get_actor(task_id, actor_id1, "id") == m_actor1.uid
    assert m_actor1.parameter_read(task_id) == "parameter2"

    # assign actor2
    m_actor2 = DistributedManager(RedisParameters(host="test"), params)
    m_actor2.set_user("actor")
    task_id_actor2, actor_id2 = m_actor2.task_assign_by_my_id()
    assert task_id_actor2 == task_id
    assert m_actor2.task_get_config(task_id) == "config"
    assert actor_id2 == 1
    assert m_actor2.task_get_actor(task_id, actor_id2, "id") == m_actor2.uid
    assert m_actor2.parameter_read(task_id) == "parameter2"

    _t, _ = m_actor2.task_assign_by_my_id()
    assert _t == ""

    # assign actor3
    m_actor3 = DistributedManager(RedisParameters(host="test"), params)
    m_actor3.set_user("actor")
    _t, _ = m_actor3.task_assign_by_my_id()
    assert _t == ""
