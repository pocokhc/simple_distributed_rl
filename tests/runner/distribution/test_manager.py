import pytest
import pytest_mock

from srl.runner.distribution.connectors.parameters import GCPParameters, RabbitMQParameters, RedisParameters
from srl.runner.distribution.manager import DistributedManager
from tests.runner.distribution.server_mock import create_gcp_mock, create_pika_mock, create_redis_mock
from tests.runner.distribution.test_connectors import memory_connector_test


def test_memory_redis(mocker: pytest_mock.MockerFixture):
    pytest.importorskip("redis")
    create_redis_mock(mocker)
    memory_connector_test(RedisParameters(host="test").create_memory_connector())


def test_memory_pika(mocker: pytest_mock.MockerFixture):
    pytest.importorskip("redis")
    create_pika_mock(mocker)
    memory_connector_test(RabbitMQParameters(host="test").create_memory_connector())


def test_memory_gcp(mocker: pytest_mock.MockerFixture):
    pytest.importorskip("redis")
    create_gcp_mock(mocker)
    memory_connector_test(GCPParameters(project_id="test").create_memory_connector())


def test_parameter(mocker: pytest_mock.MockerFixture):
    pytest.importorskip("redis")
    create_redis_mock(mocker)
    manager = DistributedManager(RedisParameters(host="test"), None)
    manager.parameter_update("a")
    assert manager.parameter_read() == "a"


@pytest.mark.parametrize("server", ["", "redis", "pika", "gcp"])
def test_no_task(mocker: pytest_mock.MockerFixture, server):
    pytest.importorskip("redis")
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
    manager.set_user("client")

    manager.task_end()
    assert manager.task_get_status() == "END"
    assert manager.task_get_actor_num() == 1
    assert manager.task_get_trainer("a") == ""
    manager.task_set_trainer("a", "a")
    assert manager.task_get_trainer("a") == "a"
    assert manager.task_get_actor(1, "a") == ""
    manager.task_set_actor(1, "a", "a")
    assert manager.task_get_actor(1, "a") == "a"
    manager.task_assign_by_my_id()
    manager.task_log("")
    manager.keepalive()


@pytest.mark.parametrize("server", ["", "redis", "pika", "gcp"])
def test_task(mocker: pytest_mock.MockerFixture, server):
    pytest.importorskip("redis")
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
    m_client.task_create(2, "config", "parameter")
    assert m_client.task_get_status() == "WAIT"
    assert m_client.task_get_actor_num() == 2
    assert m_client.task_get_trainer("id") == "NO_ASSIGN"
    assert m_client.task_get_actor(0, "id") == "NO_ASSIGN"
    assert m_client.task_get_actor(1, "id") == "NO_ASSIGN"
    assert m_client.task_get_actor(-1, "id") == ""

    # assign trainer
    m_trainer = DistributedManager(RedisParameters(host="test"), params)
    m_trainer.set_user("trainer")
    is_assigned, actor_id = m_trainer.task_assign_by_my_id()
    assert is_assigned
    assert m_trainer.task_get_config() == "config"
    assert actor_id == 0
    assert m_trainer.task_get_trainer("id") == m_trainer.uid
    assert m_trainer.parameter_read() == "parameter"
    m_trainer.parameter_update("parameter2")

    is_assigned, _ = m_trainer.task_assign_by_my_id()
    assert not is_assigned

    # assign actor1
    m_actor1 = DistributedManager(RedisParameters(host="test"), params)
    m_actor1.set_user("actor")
    is_assigned, actor_id1 = m_actor1.task_assign_by_my_id()
    assert is_assigned
    assert m_actor1.task_get_config() == "config"
    assert actor_id1 == 0
    assert m_actor1.task_get_actor(actor_id1, "id") == m_actor1.uid
    assert m_actor1.parameter_read() == "parameter2"

    # assign actor2
    m_actor2 = DistributedManager(RedisParameters(host="test"), params)
    m_actor2.set_user("actor")
    is_assigned, actor_id2 = m_actor2.task_assign_by_my_id()
    assert is_assigned
    assert m_actor2.task_get_config() == "config"
    assert actor_id2 == 1
    assert m_actor2.task_get_actor(actor_id2, "id") == m_actor2.uid
    assert m_actor2.parameter_read() == "parameter2"

    is_assigned, _ = m_actor2.task_assign_by_my_id()
    assert not is_assigned

    # assign actor3
    m_actor3 = DistributedManager(RedisParameters(host="test"), params)
    m_actor3.set_user("actor")
    is_assigned, _ = m_actor3.task_assign_by_my_id()
    assert not is_assigned
