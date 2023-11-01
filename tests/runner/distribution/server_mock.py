from queue import Queue
from typing import Dict, Optional, Union
from unittest.mock import Mock

import pika.exceptions
import pytest_mock

from srl.runner.distribution import manager as manager_module


class PikaMock(Mock):
    queues: Dict[str, Queue] = {}

    def __init__(self):
        super().__init__()
        self.connection = False

    def channel(self):
        self.connection = True
        return self

    def close(self):
        self.connection = False

    def basic_publish(self, exchange: str, routing_key: str, body: Union[str, bytes]):
        if not self.connection:
            raise pika.exceptions.ChannelClosedByBroker(403, "NOT_CONNECTED")
        # なくても例外は出ない
        if routing_key not in self.queues:
            return
        self.queues[routing_key].put(body)

    def basic_get(self, queue: str, auto_ack: bool = False):
        if not self.connection:
            raise pika.exceptions.ChannelClosedByBroker(403, "NOT_CONNECTED")
        # ない場合は例外
        if queue not in self.queues:
            self.connection = False
            raise pika.exceptions.ChannelClosedByBroker(404, "NOT_FOUND")
        if self.queues[queue].empty():
            return None, None, None
        return {}, {}, self.queues[queue].get()

    def queue_declare(self, queue: str, passive: bool = False):
        if not self.connection:
            raise pika.exceptions.ChannelClosedByBroker(403, "NOT_CONNECTED")
        # ない場合は例外
        if passive:
            if queue not in self.queues:
                self.connection = False
                raise pika.exceptions.ChannelClosedByBroker(404, "NOT_FOUND")
        else:
            if queue not in self.queues:
                self.queues[queue] = Queue()

        mock = Mock()
        mock.method.message_count = self.queues[queue].qsize()
        return mock


class RedisMock(Mock):
    tbl = {}
    queues: Dict[str, Queue] = {}

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
            self.queues[key] = Queue()
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


def create_mock(mocker: pytest_mock.MockerFixture):
    # --- redis
    mock_redis_cls = mocker.patch.object(manager_module.redis, "Redis", autospec=True)
    mock_redis = RedisMock()
    mock_redis_cls.return_value = mock_redis

    # --- pika
    mock_pika_cls = mocker.patch.object(manager_module.pika, "BlockingConnection", autospec=True)
    mock_pika = PikaMock()
    mock_pika_cls.return_value = mock_pika

    return mock_redis, mock_pika
