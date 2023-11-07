import logging
import pickle
import time
import traceback
from typing import Any, Optional, cast

import google.api_core.exceptions
from google.cloud import pubsub_v1

from srl.runner.distribution.connectors.imemory import IMemoryConnector
from srl.runner.distribution.connectors.parameters import GCPParameters

logger = logging.getLogger(__name__)


class GCPPubSubConnector(IMemoryConnector):
    def __init__(self, parameter: GCPParameters):
        self.parameter = parameter
        self.publisher = None
        self.subscriber = None

    def __del__(self):
        self.close()

    def close(self):
        self.publisher = None
        self.subscriber = None

    def connect_publisher(self) -> bool:
        if self.publisher is not None:
            return True
        try:
            self.publisher = pubsub_v1.PublisherClient()
            return True
        except Exception:
            logger.error(traceback.format_exc())
            self.publisher = None
        return False

    def connect_subscriber(self) -> bool:
        if self.subscriber is not None:
            return True
        try:
            self.subscriber = pubsub_v1.SubscriberClient()
            return True
        except Exception:
            logger.error(traceback.format_exc())
            self.subscriber = None
        return False

    def ping(self) -> bool:
        self.connect_publisher()
        self.connect_subscriber()
        return True

    def _create_memory_name(self, task_id: str):
        return f"task:{task_id}".replace(":", "_")

    def memory_setup(self, task_id: str):
        self.memory_name = self._create_memory_name(task_id)
        self._close_setup()

    def _close_setup(self):
        self.close()
        self.connect_publisher()
        self.connect_subscriber()
        assert self.publisher is not None
        assert self.subscriber is not None

        self.topic_path = self.publisher.topic_path(self.parameter.project_id, self.memory_name)
        self.subscription_path = self.subscriber.subscription_path(self.parameter.project_id, self.memory_name)

        try:
            self.publisher.create_topic(name=self.topic_path)
            logger.info(f"create topic {self.memory_name}")
        except google.api_core.exceptions.AlreadyExists:
            pass

        try:
            self.subscriber.create_subscription(name=self.subscription_path, topic=self.topic_path)
            logger.info(f"create subscription {self.memory_name}")
        except google.api_core.exceptions.AlreadyExists:
            pass

        # topicが出来るまで待機
        topic = self.publisher.get_topic(topic=self.topic_path)
        for _ in range(120):
            if topic is not None:
                break
            print(f"Topic {self.memory_name} is not available yet. Waiting...")
            time.sleep(1)
            topic = self.publisher.get_topic(topic=self.topic_path)

        subscription = self.subscriber.get_subscription(subscription=self.subscription_path)
        for _ in range(120):
            if subscription is not None:
                break
            print(f"Subscription {self.memory_name} is not available yet. Waiting...")
            time.sleep(1)
            subscription = self.subscriber.get_subscription(subscription=self.subscription_path)

    def memory_add(self, dat: Any) -> bool:
        try:
            assert self.publisher is not None
            dat = pickle.dumps(dat)
            self.publisher.publish(self.topic_path, dat)
            return True
        except Exception:
            logger.error(traceback.format_exc())
            self._close_setup()
        return False

    def memory_recv(self) -> Optional[Any]:
        try:
            assert self.subscriber is not None

            response = self.subscriber.pull(
                subscription=self.subscription_path, max_messages=1, return_immediately=True
            )
            if len(response.received_messages) == 0:
                return None
            mess = response.received_messages[0]
            self.subscriber.acknowledge(subscription=self.subscription_path, ack_ids=[mess.ack_id])
            dat = mess.message.data
            return dat if dat is None else pickle.loads(cast(bytes, dat))
        except Exception:
            logger.error(traceback.format_exc())
            self._close_setup()
        return None

    def memory_size(self) -> int:
        return -1

    def memory_delete_if_exist(self, task_id: str) -> None:
        name = self._create_memory_name(task_id)

        assert self.subscriber
        try:
            subscription_path = self.subscriber.subscription_path(self.parameter.project_id, name)
            self.subscriber.delete_subscription(subscription=subscription_path)
            logger.info(f"delete subscription {name}")
        except google.api_core.exceptions.NotFound as e:
            logger.info(e)

        assert self.publisher
        try:
            topic_path = self.publisher.topic_path(self.parameter.project_id, name)
            self.publisher.delete_topic(topic=topic_path)
            logger.info(f"delete topic {name}")
        except google.api_core.exceptions.NotFound as e:
            logger.info(e)

    def memory_exist(self, task_id: str) -> bool:
        # どちらかがあればあるとする
        name = self._create_memory_name(task_id)

        assert self.subscriber
        try:
            subscription = self.subscriber.subscription_path(self.parameter.project_id, name)
            self.subscriber.get_subscription(subscription=subscription)
            return True
        except google.api_core.exceptions.NotFound as e:
            logger.info(e)

        assert self.publisher
        try:
            topic_path = self.publisher.topic_path(self.parameter.project_id, name)
            self.publisher.get_topic(topic=topic_path)
            return True
        except google.api_core.exceptions.NotFound as e:
            logger.info(e)

        return False
