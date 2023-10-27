import queue
from typing import Dict, Optional
from unittest.mock import Mock

import pytest
import pytest_mock

from srl.runner.distribution import manager as manager_module
from srl.runner.distribution.manager import DistributedManager


class RedisMock(Mock):
    tbl = {}
    queues: Dict[str, queue.Queue] = {}

    def get(self, key: str) -> Optional[bytes]:
        return self.tbl.get(key, None)

    def set(self, key: str, value):
        # どうやら文字列もバイナリになる
        if isinstance(value, str):
            value = value.encode()
        self.tbl[key] = value
        return True

    def exists(self, key: str) -> bool:
        return key in self.tbl

    def keys(self, filter: str):
        filter = filter.replace("*", "")
        keys = [k for k in list(self.tbl.keys()) if filter in k]
        return keys

    def rpush(self, key: str, value):
        if key not in self.queues:
            self.queues[key] = queue.Queue()
        self.queues[key].put(value)
        return True

    def lpop(self, key: str):
        if key not in self.queues:
            return None
        if self.queues[key].empty():
            return None
        return self.queues[key].get(timeout=1)

    def llen(self, key: str):
        if key not in self.queues:
            return 0
        return self.queues[key].qsize()


def _create_mock(mocker: pytest_mock.MockerFixture):
    mock_redis_cls = mocker.patch.object(manager_module.redis, "Redis", autospec=True)
    mock_redis = RedisMock()
    mock_redis_cls.return_value = mock_redis


def test_memory(mocker: pytest_mock.MockerFixture):
    _create_mock(mocker)
    manager = DistributedManager("test")

    manager.memory_add("0", "a")
    manager.memory_add("1", "b")

    assert manager.memory_size("0") == 1
    assert manager.memory_size("1") == 1
    assert manager.memory_size("2") == 0

    assert manager.memory_recv("0") == "a"
    assert manager.memory_recv("0") is None

    manager.memory_reset("1")
    assert manager.memory_size("1") == 0


def test_parameter(mocker: pytest_mock.MockerFixture):
    _create_mock(mocker)
    manager = DistributedManager("test")
    manager.set_user("trainer")

    assert manager.parameter_read("0") is None

    # 自分のidが登録されている場合のみ更新
    with pytest.raises(AssertionError):
        manager.parameter_update("0", "a")


def test_no_task(mocker: pytest_mock.MockerFixture):
    _create_mock(mocker)
    manager = DistributedManager("test")

    manager.task_end("0")
    assert manager.task_get_status("0") == ""
    assert manager.task_get_actor_num("0") == -1
    assert manager.task_get_trainer("0", "a") == ""
    manager.task_set_trainer("0", "a", "a")
    assert manager.task_get_actor("0", 1, "a") == ""
    manager.task_set_actor("0", 1, "a", "a")
    manager.task_assign_by_my_id()
    manager.log("0", "")
    manager.keepalive("0")


def test_task(mocker: pytest_mock.MockerFixture):
    _create_mock(mocker)

    manager = DistributedManager("test")
    manager.set_user("client")
    task_id = manager.task_add(2, "config", "parameter")
    assert manager.task_get_status(task_id) == "WAIT"
    assert manager.task_get_actor_num(task_id) == 2
    assert manager.task_get_trainer(task_id, "id") == "NO_ASSIGN"
    assert manager.task_get_actor(task_id, 0, "id") == "NO_ASSIGN"
    assert manager.task_get_actor(task_id, 1, "id") == "NO_ASSIGN"
    assert manager.task_get_actor(task_id, -1, "id") == ""

    # assign trainer
    manager_trainer = DistributedManager("test")
    manager_trainer.set_user("trainer")
    task_id_trainer, config, actor_id = manager_trainer.task_assign_by_my_id()
    assert task_id_trainer == task_id
    assert config == "config"
    assert actor_id == 0
    assert manager_trainer.task_get_trainer(task_id, "id") == manager_trainer.uid
    assert manager_trainer.parameter_read(task_id) == "parameter"
    manager_trainer.parameter_update(task_id, "parameter2")

    _, config, _ = manager_trainer.task_assign_by_my_id()
    assert config is None

    # assign actor1
    manager_actor1 = DistributedManager("test")
    manager_actor1.set_user("actor")
    task_id_actor1, config, actor_id1 = manager_actor1.task_assign_by_my_id()
    assert task_id_actor1 == task_id
    assert config == "config"
    assert actor_id1 == 0
    assert manager_actor1.task_get_actor(task_id, actor_id1, "id") == manager_actor1.uid
    assert manager_actor1.parameter_read(task_id) == "parameter2"

    # assign actor2
    manager_actor2 = DistributedManager("test")
    manager_actor2.set_user("actor")
    task_id_actor2, config, actor_id2 = manager_actor2.task_assign_by_my_id()
    assert task_id_actor2 == task_id
    assert config == "config"
    assert actor_id2 == 1
    assert manager_actor2.task_get_actor(task_id, actor_id2, "id") == manager_actor2.uid
    assert manager_actor2.parameter_read(task_id) == "parameter2"

    _, config, _ = manager_actor2.task_assign_by_my_id()
    assert config is None

    # assign actor3
    manager_actor3 = DistributedManager("test")
    manager_actor3.set_user("actor")
    _, config, _ = manager_actor3.task_assign_by_my_id()
    assert config is None
