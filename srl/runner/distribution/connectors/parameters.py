from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from srl.runner.distribution.interface import IMemoryReceiver, IMemorySender, IMemoryServerParameters

if TYPE_CHECKING:
    from .redis_ import RedisConnector


@dataclass
class RedisParameters(IMemoryServerParameters):
    url: str = ""
    host: str = ""
    port: int = 6379
    db: int = 0
    kwargs: dict = field(default_factory=dict)
    task_name: str = "task"
    parameter_name: str = "task:raw:parameter"
    queue_name: str = "task:raw:mq"

    def _create(self):
        assert self.url != "" or self.host != "", "Please specify 'host' or 'url'."
        from .redis_ import RedisConnector

        return RedisConnector(self)

    def create_connector(self) -> "RedisConnector":
        return self._create()

    def create_memory_receiver(self) -> IMemoryReceiver:
        return self._create()

    def create_memory_sender(self) -> IMemorySender:
        return self._create()


@dataclass
class RabbitMQParameters(IMemoryServerParameters):
    url: str = ""
    host: str = ""
    port: int = 5672
    username: str = "guest"
    password: str = "guest"
    virtual_host: str = "/"
    ssl: bool = True
    kwargs: dict = field(default_factory=dict)
    queue_name: str = "mq"

    def create_memory_receiver(self) -> IMemoryReceiver:
        assert self.url != "" or self.host != "", "Please specify 'host' or 'url'."
        from .rabbitmq import RabbitMQReceiver

        return RabbitMQReceiver(self)

    def create_memory_sender(self) -> IMemorySender:
        assert self.url != "" or self.host != "", "Please specify 'host' or 'url'."
        from .rabbitmq import RabbitMQSender

        return RabbitMQSender(self)


@dataclass
class MQTTParameters(IMemoryServerParameters):
    host: str
    port: int = 1883
    kwargs: dict = field(default_factory=dict)
    topic_name: str = "mq"

    def create_memory_receiver(self) -> IMemoryReceiver:
        from .mqtt import MQTTReceiver

        return MQTTReceiver(self)

    def create_memory_sender(self) -> IMemorySender:
        from .mqtt import MQTTSender

        return MQTTSender(self)


@dataclass
class GCPParameters(IMemoryServerParameters):
    project_id: str
    topic_name: str = "mq"
    subscription_name: str = "mq"

    def create_memory_receiver(self) -> IMemoryReceiver:
        from .gcp import GCPPubSubReceiver

        return GCPPubSubReceiver(self)

    def create_memory_sender(self) -> IMemorySender:
        from .gcp import GCPPubSubSender

        return GCPPubSubSender(self)
