from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


class IServerConnector(ABC):
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Serverへの接続無し。例外は出さない"""
        raise NotImplementedError()

    @abstractmethod
    def ping(self) -> bool:
        """Serverへ接続状況を問い合わせる。例外は出さない"""
        raise NotImplementedError()


# --- parameter
class IParameterServer(IServerConnector):
    @abstractmethod
    def parameter_write(self, param_dat: Optional[Any], init: bool = False):
        """例外を出す"""
        raise NotImplementedError()

    @abstractmethod
    def parameter_read(self) -> Optional[Any]:
        """例外を出す"""
        raise NotImplementedError()


# --- memory
class IMemoryReceiver(IServerConnector):
    @abstractmethod
    def memory_recv(self) -> Optional[Any]:
        """例外を出す"""
        raise NotImplementedError()

    @abstractmethod
    def memory_purge(self) -> None:
        """例外は出さない"""
        raise NotImplementedError()


class IMemorySender(IServerConnector):
    @abstractmethod
    def memory_send(self, dat: Any) -> None:
        """例外を出す"""
        raise NotImplementedError()

    @abstractmethod
    def memory_size(self) -> int:
        """例外は出さない、失敗時は-1"""
        raise NotImplementedError()


@dataclass
class IMemoryServerParameters(ABC):
    @abstractmethod
    def create_memory_receiver(self) -> IMemoryReceiver:
        raise NotImplementedError()

    @abstractmethod
    def create_memory_sender(self) -> IMemorySender:
        raise NotImplementedError()


@dataclass
class RedisParameters(IMemoryServerParameters):
    url: str = ""
    host: str = ""
    port: int = 6379
    db: int = 0
    redis_kwargs: dict = field(default_factory=dict)
    # task
    task_key: str = "task"
    parameter_key: str = "raw:parameter"
    queue_key: str = "raw:mq"

    def create(self):
        assert self.url != "" or self.host != "", "Please specify 'host' or 'url'."
        from .connectors.redis_ import RedisConnector

        return RedisConnector(self)

    def create_memory_receiver(self) -> IMemoryReceiver:
        return self.create()

    def create_memory_sender(self) -> IMemorySender:
        return self.create()


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
        from .connectors.rabbitmq import RabbitMQReceiver

        return RabbitMQReceiver(self)

    def create_memory_sender(self) -> IMemorySender:
        assert self.url != "" or self.host != "", "Please specify 'host' or 'url'."
        from .connectors.rabbitmq import RabbitMQSender

        return RabbitMQSender(self)


@dataclass
class MQTTParameters(IMemoryServerParameters):
    host: str
    port: int = 1883
    kwargs: dict = field(default_factory=dict)
    topic_name: str = "mq"

    def create_memory_receiver(self) -> IMemoryReceiver:
        from .connectors.mqtt import MQTTReceiver

        return MQTTReceiver(self)

    def create_memory_sender(self) -> IMemorySender:
        from .connectors.mqtt import MQTTSender

        return MQTTSender(self)


@dataclass
class GCPParameters(IMemoryServerParameters):
    project_id: str
    topic_name: str = "mq"
    subscription_name: str = "mq"

    def create_memory_receiver(self) -> IMemoryReceiver:
        from .connectors.gcp import GCPPubSubReceiver

        return GCPPubSubReceiver(self)

    def create_memory_sender(self) -> IMemorySender:
        from .connectors.gcp import GCPPubSubSender

        return GCPPubSubSender(self)
