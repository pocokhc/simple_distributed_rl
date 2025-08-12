import logging
import pickle
import traceback
from typing import Any, Optional, cast

from google.cloud import pubsub_v1

from srl.runner.distribution.connector_configs import GCPParameters, IMemoryReceiver, IMemorySender

logger = logging.getLogger(__name__)


class GCPPubSubReceiver(IMemoryReceiver):
    def __init__(self, parameter: GCPParameters):
        self.parameter = parameter
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        self.subscriber = None
        self.subscription_path = ""

    def _connect_subscriber(self) -> bool:
        if self.subscriber is not None:
            return True
        try:
            self.subscriber = pubsub_v1.SubscriberClient()
            self.subscription_path = self.subscriber.subscription_path(self.parameter.project_id, self.parameter.subscription_name)

            # try:
            #     self.subscriber.create_subscription(name=self.subscription_path, topic=self.topic_path)
            #     logger.info(f"create subscription {self.parameter.subscription_name}")
            #     subscription = self.subscriber.get_subscription(subscription=self.subscription_path)
            #     for _ in range(120):
            #         if subscription is not None:
            #             break
            #         print("Subscription is not available yet. Waiting...")
            #         time.sleep(1)
            #         subscription = self.subscriber.get_subscription(subscription=self.subscription_path)
            # except google.api_core.exceptions.AlreadyExists:
            #     pass

            return True
        except Exception:
            logger.error(traceback.format_exc())
            self.subscriber = None
        return False

    @property
    def is_connected(self) -> bool:
        return self.subscriber is not None

    def ping(self) -> bool:
        try:
            self._connect_subscriber()
            assert self.subscriber is not None
            subscription = self.subscriber.get_subscription(subscription=self.subscription_path)
            return subscription is not None
        except Exception as e:
            logger.error(e)
        return False

    def memory_recv(self) -> Optional[Any]:
        try:
            if self.subscriber is None:
                self._connect_subscriber()
                assert self.subscriber is not None
            response = self.subscriber.pull(subscription=self.subscription_path, max_messages=1, return_immediately=True)
            if len(response.received_messages) == 0:
                return None
            mess = response.received_messages[0]
            self.subscriber.acknowledge(subscription=self.subscription_path, ack_ids=[mess.ack_id])
            dat = mess.message.data
            return dat if dat is None else pickle.loads(cast(bytes, dat))
        except Exception:
            self.close()
            raise

    def memory_purge(self) -> None:
        try:
            if self.subscriber is None:
                self._connect_subscriber()
                assert self.subscriber is not None
            for _ in range(100):
                response = self.subscriber.pull(subscription=self.subscription_path, max_messages=100, return_immediately=True)
                received_messages = response.received_messages
                if not received_messages:
                    break

                ack_ids = [message.ack_id for message in received_messages]
                self.subscriber.acknowledge(subscription=self.subscription_path, ack_ids=ack_ids)
        except Exception:
            logger.error(traceback.format_exc())
            self.close()


class GCPPubSubSender(IMemorySender):
    def __init__(self, parameter: GCPParameters):
        self.parameter = parameter
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        self.publisher = None
        self.topic_path = ""

    def _connect_publisher(self) -> bool:
        if self.publisher is not None:
            return True
        try:
            self.publisher = pubsub_v1.PublisherClient()
            self.topic_path = self.publisher.topic_path(self.parameter.project_id, self.parameter.topic_name)

            # try:
            #     self.publisher.create_topic(name=self.topic_path)
            #     logger.info(f"create topic {self.parameter.topic_name}")
            #     topic = self.publisher.get_topic(topic=self.topic_path)
            #     for _ in range(120):
            #         if topic is not None:
            #             break
            #         print("Topic is not available yet. Waiting...")
            #         time.sleep(1)
            #         topic = self.publisher.get_topic(topic=self.topic_path)
            # except google.api_core.exceptions.AlreadyExists:
            #     pass

            return True
        except Exception:
            logger.error(traceback.format_exc())
            self.publisher = None
        return False

    @property
    def is_connected(self) -> bool:
        return self.publisher is not None

    def ping(self) -> bool:
        try:
            self._connect_publisher()
            assert self.publisher is not None
            topic = self.publisher.get_topic(topic=self.topic_path)
            return topic is not None
        except Exception as e:
            logger.error(e)
        return False

    def memory_send(self, dat: Any) -> None:
        try:
            if self.publisher is None:
                self._connect_publisher()
                assert self.publisher is not None
            dat = pickle.dumps(dat)
            self.publisher.publish(self.topic_path, dat)
        except Exception:
            self.close()
            raise

    def memory_size(self) -> int:
        return -1
