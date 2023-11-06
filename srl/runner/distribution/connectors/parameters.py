from dataclasses import dataclass, field

from srl.runner.distribution.connectors.imemory import IMemoryConnector, IServerParameters


@dataclass
class RedisParameters(IServerParameters):
    url: str = ""
    host: str = ""
    kwargs: dict = field(default_factory=dict)

    def create_memory_connector(self) -> "IMemoryConnector":
        from .redis_ import RedisConnector

        return RedisConnector(self)


@dataclass
class RabbitMQParameters(IServerParameters):
    url: str = ""
    host: str = ""
    port: int = 5672
    username: str = "guest"
    password: str = "guest"
    virtual_host: str = "/"
    ssl: bool = True
    kwargs: dict = field(default_factory=dict)

    def create_memory_connector(self) -> IMemoryConnector:
        from .rabbitmq import RabbitMQConnector

        return RabbitMQConnector(self)


@dataclass
class GCPParameters(IServerParameters):
    project_id: str = ""

    def create_memory_connector(self) -> IMemoryConnector:
        from .gcp import GCPubSubConnector

        return GCPubSubConnector(self)
