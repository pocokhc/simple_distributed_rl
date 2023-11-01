import pytest
import pytest_mock

from srl.runner.distribution.manager import DistributedManager, ServerParameters
from tests.runner.distribution.server_mock import create_mock


@pytest.mark.parametrize("server", ["redis", "pika"])
def test_memory(mocker: pytest_mock.MockerFixture, server):
    create_mock(mocker)
    if server == "redis":
        rabbitmq_host = ""
    else:
        rabbitmq_host = "test"
    manager = DistributedManager(ServerParameters(rabbitmq_host=rabbitmq_host))

    m = manager.create_memory_connector()
    assert m.ping()

    m.memory_add("0", {"a": 1})
    m.memory_add("1", {"b": 2})

    assert m.memory_size("0") == 1
    assert m.memory_size("1") == 1
    assert m.memory_size("2") == 0

    assert m.memory_recv("0") == {"a": 1}
    assert m.memory_recv("0") is None


def test_parameter(mocker: pytest_mock.MockerFixture):
    create_mock(mocker)
    manager = DistributedManager(ServerParameters())
    manager.set_user("trainer")

    assert manager.parameter_read("0") is None

    # 自分のidが登録されている場合のみ更新
    with pytest.raises(AssertionError):
        manager.parameter_update("0", "a")


@pytest.mark.parametrize("server", ["redis", "pika"])
def test_no_task(mocker: pytest_mock.MockerFixture, server):
    create_mock(mocker)
    if server == "redis":
        rabbitmq_host = ""
    else:
        rabbitmq_host = "test"
    manager = DistributedManager(ServerParameters(rabbitmq_host=rabbitmq_host))

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


@pytest.mark.parametrize("server", ["redis", "pika"])
def test_task(mocker: pytest_mock.MockerFixture, server):
    create_mock(mocker)
    if server == "redis":
        rabbitmq_host = ""
    else:
        rabbitmq_host = "test"

    manager = DistributedManager(ServerParameters(rabbitmq_host=rabbitmq_host))
    manager.set_user("client")
    task_id = manager.task_add(2, "config", "parameter")
    assert manager.task_get_status(task_id) == "WAIT"
    assert manager.task_get_actor_num(task_id) == 2
    assert manager.task_get_trainer(task_id, "id") == "NO_ASSIGN"
    assert manager.task_get_actor(task_id, 0, "id") == "NO_ASSIGN"
    assert manager.task_get_actor(task_id, 1, "id") == "NO_ASSIGN"
    assert manager.task_get_actor(task_id, -1, "id") == ""

    # assign trainer
    manager_trainer = DistributedManager(ServerParameters(rabbitmq_host=rabbitmq_host))
    manager_trainer.set_user("trainer")
    task_id_trainer, actor_id = manager_trainer.task_assign_by_my_id()
    assert task_id_trainer == task_id
    assert manager_trainer.task_get_config(task_id) == "config"
    assert actor_id == 0
    assert manager_trainer.task_get_trainer(task_id, "id") == manager_trainer.uid
    assert manager_trainer.parameter_read(task_id) == "parameter"
    manager_trainer.parameter_update(task_id, "parameter2")

    _t, _ = manager_trainer.task_assign_by_my_id()
    assert _t == ""

    # assign actor1
    manager_actor1 = DistributedManager(ServerParameters(rabbitmq_host=rabbitmq_host))
    manager_actor1.set_user("actor")
    task_id_actor1, actor_id1 = manager_actor1.task_assign_by_my_id()
    assert task_id_actor1 == task_id
    assert manager_actor1.task_get_config(task_id) == "config"
    assert actor_id1 == 0
    assert manager_actor1.task_get_actor(task_id, actor_id1, "id") == manager_actor1.uid
    assert manager_actor1.parameter_read(task_id) == "parameter2"

    # assign actor2
    manager_actor2 = DistributedManager(ServerParameters(rabbitmq_host=rabbitmq_host))
    manager_actor2.set_user("actor")
    task_id_actor2, actor_id2 = manager_actor2.task_assign_by_my_id()
    assert task_id_actor2 == task_id
    assert manager_actor2.task_get_config(task_id) == "config"
    assert actor_id2 == 1
    assert manager_actor2.task_get_actor(task_id, actor_id2, "id") == manager_actor2.uid
    assert manager_actor2.parameter_read(task_id) == "parameter2"

    _t, _ = manager_actor2.task_assign_by_my_id()
    assert _t == ""

    # assign actor3
    manager_actor3 = DistributedManager(ServerParameters(rabbitmq_host=rabbitmq_host))
    manager_actor3.set_user("actor")
    _t, _ = manager_actor3.task_assign_by_my_id()
    assert _t == ""
