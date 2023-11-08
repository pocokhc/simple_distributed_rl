from queue import Queue
from typing import Dict, Optional, Union
from unittest.mock import Mock

try:
    import google.api_core.exceptions
    import pika.exceptions
except ModuleNotFoundError as e:
    print(e)

import pytest_mock


class PikaMock:
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
        if routing_key not in PikaMock.queues:
            return
        PikaMock.queues[routing_key].put(body)

    def basic_get(self, queue: str, auto_ack: bool = False):
        if not self.connection:
            raise pika.exceptions.ChannelClosedByBroker(403, "NOT_CONNECTED")
        # ない場合は例外
        if queue not in PikaMock.queues:
            self.connection = False
            raise pika.exceptions.ChannelClosedByBroker(404, "NOT_FOUND")
        if PikaMock.queues[queue].empty():
            return None, None, None
        return {}, {}, PikaMock.queues[queue].get()

    def queue_declare(self, queue: str, passive: bool = False):
        if not self.connection:
            raise pika.exceptions.ChannelClosedByBroker(403, "NOT_CONNECTED")
        # ない場合は例外
        if passive:
            if queue not in PikaMock.queues:
                self.connection = False
                raise pika.exceptions.ChannelClosedByBroker(404, "NOT_FOUND")
        else:
            if queue not in PikaMock.queues:
                PikaMock.queues[queue] = Queue()

        mock = Mock()
        mock.method.message_count = PikaMock.queues[queue].qsize()
        return mock

    def queue_delete(self, queue: str):
        if not self.connection:
            raise pika.exceptions.ChannelClosedByBroker(403, "NOT_CONNECTED")
        # ない場合は例外
        if queue not in PikaMock.queues:
            self.connection = False
            raise pika.exceptions.ChannelClosedByBroker(404, "NOT_FOUND")
        del PikaMock.queues[queue]

    def queue_purge(self, queue: str):
        if not self.connection:
            raise pika.exceptions.ChannelClosedByBroker(403, "NOT_CONNECTED")
        # ない場合は例外
        if queue not in PikaMock.queues:
            self.connection = False
            raise pika.exceptions.ChannelClosedByBroker(404, "NOT_FOUND")
        PikaMock.queues[queue] = Queue()


class RedisMock:
    tbl = {}
    queues: Dict[str, Queue] = {}

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        pass

    def get(self, key: str) -> Optional[bytes]:
        return RedisMock.tbl.get(key, None)

    def set(self, key: str, value):
        # どうやら文字列もバイナリになる
        if isinstance(value, str):
            value = value.encode()
        RedisMock.tbl[key] = value
        return True

    def exists(self, key: str) -> bool:
        if key in RedisMock.tbl:
            return True
        return key in RedisMock.queues

    def delete(self, key: str):
        del RedisMock.queues[key]

    def keys(self, filter: str):
        filter = filter.replace("*", "")
        keys = [k for k in list(RedisMock.tbl.keys()) if filter in k]
        return keys

    def scan_iter(self, filter: str):
        return self.keys(filter)

    def rpush(self, key: str, value):
        if key not in RedisMock.queues:
            RedisMock.queues[key] = Queue()
        RedisMock.queues[key].put(value)
        return True

    def lpop(self, key: str):
        if key not in RedisMock.queues:
            return None
        if RedisMock.queues[key].empty():
            return None
        return RedisMock.queues[key].get(timeout=1)

    def llen(self, key: str):
        if key not in RedisMock.queues:
            return 0
        return RedisMock.queues[key].qsize()


class GCPMock:
    tbl = {}
    queues: Dict[str, Queue] = {}

    def topic_path(self, project: str, topic: str):
        return f"{project}/{topic}"

    def subscription_path(self, project: str, subscription: str):
        return f"{project}/{subscription}"

    def create_topic(self, name: str):
        if name not in GCPMock.queues:
            GCPMock.queues[name] = Queue()

    def create_subscription(self, name: str, topic: str):
        GCPMock.tbl[name] = topic
        if topic not in GCPMock.queues:
            GCPMock.queues[topic] = Queue()

    def get_topic(self, topic: str):
        if topic not in GCPMock.queues:
            raise google.api_core.exceptions.NotFound("Not found")
        return {}

    def get_subscription(self, subscription: str):
        if subscription not in GCPMock.tbl:
            raise google.api_core.exceptions.NotFound("Not found")
        return {}

    def publish(self, topic: str, dat):
        GCPMock.queues[topic].put(dat)

    def pull(self, subscription: str, **kwargs):
        if GCPMock.queues[GCPMock.tbl[subscription]].empty():
            mock = Mock()
            mock.received_messages = []
            return mock
        dat = GCPMock.queues[GCPMock.tbl[subscription]].get(timeout=1)

        mock2 = Mock()
        mock2.ack_id = 1
        mock2.message.data = dat

        mock = Mock()
        mock.received_messages = [mock2]

        return mock

    def acknowledge(self, subscription: str, **kwargs):
        pass

    def delete_topic(self, topic: str):
        if topic not in GCPMock.queues:
            raise google.api_core.exceptions.NotFound("Not found")
        del GCPMock.queues[topic]

    def delete_subscription(self, subscription: str):
        if subscription not in GCPMock.tbl:
            raise google.api_core.exceptions.NotFound("Not found")
        del GCPMock.queues[GCPMock.tbl[subscription]]


def create_redis_mock(mocker: pytest_mock.MockerFixture):
    from srl.runner.distribution.connectors import redis_

    mock_redis_cls = mocker.patch.object(redis_.redis, "Redis", autospec=True)
    mock_redis = RedisMock()
    mock_redis_cls.return_value = mock_redis
    return mock_redis


def create_pika_mock(mocker: pytest_mock.MockerFixture):
    from srl.runner.distribution.connectors import rabbitmq

    mock_pika_cls = mocker.patch.object(rabbitmq.pika, "BlockingConnection", autospec=True)
    mock_pika = PikaMock()
    mock_pika_cls.return_value = mock_pika
    return mock_pika


def create_gcp_mock(mocker: pytest_mock.MockerFixture):
    from srl.runner.distribution.connectors import gcp

    mock_gcp = GCPMock()
    mock_gcp_cls = mocker.patch.object(gcp.pubsub_v1, "PublisherClient", autospec=True)
    mock_gcp_cls.return_value = mock_gcp
    mock_gcp_cls = mocker.patch.object(gcp.pubsub_v1, "SubscriberClient", autospec=True)
    mock_gcp_cls.return_value = mock_gcp

    return mock_gcp
