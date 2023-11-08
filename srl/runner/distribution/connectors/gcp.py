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
        self._setup()

    def __del__(self):
        self.close()

    def close(self):
        self.publisher = None
        self.subscriber = None

    def _connect_publisher(self) -> bool:
        if self.publisher is not None:
            return True
        try:
            self.publisher = pubsub_v1.PublisherClient()
            return True
        except Exception:
            logger.error(traceback.format_exc())
            self.publisher = None
        return False

    def _connect_subscriber(self) -> bool:
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
        self._setup()
        return True

    def _setup(self):
        self._connect_publisher()
        self._connect_subscriber()
        assert self.publisher is not None
        assert self.subscriber is not None

        self.topic_path = self.publisher.topic_path(self.parameter.project_id, self.parameter.topic_name)
        self.subscription_path = self.subscriber.subscription_path(
            self.parameter.project_id, self.parameter.subscription_name
        )

        try:
            self.publisher.create_topic(name=self.topic_path)
            logger.info(f"create topic {self.parameter.topic_name}")
        except google.api_core.exceptions.AlreadyExists:
            pass

        try:
            self.subscriber.create_subscription(name=self.subscription_path, topic=self.topic_path)
            logger.info(f"create subscription {self.parameter.subscription_name}")
        except google.api_core.exceptions.AlreadyExists:
            pass

        # topicが出来るまで待機
        topic = self.publisher.get_topic(topic=self.topic_path)
        for _ in range(120):
            if topic is not None:
                break
            print("Topic is not available yet. Waiting...")
            time.sleep(1)
            topic = self.publisher.get_topic(topic=self.topic_path)

        subscription = self.subscriber.get_subscription(subscription=self.subscription_path)
        for _ in range(120):
            if subscription is not None:
                break
            print("Subscription is not available yet. Waiting...")
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
            self.close()
            self._setup()
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
            self.close()
            self._setup()
        return None

    def memory_size(self) -> int:
        return -1

    def memory_purge(self) -> None:
        try:
            assert self.subscriber
            for _ in range(100):
                response = self.subscriber.pull(
                    subscription=self.subscription_path, max_messages=100, return_immediately=True
                )
                received_messages = response.received_messages
                if not received_messages:
                    break

                ack_ids = [message.ack_id for message in received_messages]
                self.subscriber.acknowledge(subscription=self.subscription_path, ack_ids=ack_ids)

        except Exception:
            logger.error(traceback.format_exc())
            self.close()
            self._setup()
