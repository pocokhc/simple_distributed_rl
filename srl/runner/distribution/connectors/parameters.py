from dataclasses import dataclass, field

from srl.runner.distribution.connectors.imemory import IMemoryConnector, IServerParameters


@dataclass
class RedisParameters(IServerParameters):
    url: str = ""
    host: str = ""
    port: int = 6379
    db: int = 0
    kwargs: dict = field(default_factory=dict)
    task_name: str = "task"
    queue_name: str = "mq"

    def create_memory_connector(self) -> "IMemoryConnector":
        assert self.url != "" or self.host != "", "Please specify 'host' or 'url'."
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
    queue_name: str = "mq"

    def create_memory_connector(self) -> IMemoryConnector:
        assert self.url != "" or self.host != "", "Please specify 'host' or 'url'."
        from .rabbitmq import RabbitMQConnector

        return RabbitMQConnector(self)


@dataclass
class GCPParameters(IServerParameters):
    project_id: str = ""
    topic_name: str = "mq"
    subscription_name: str = "mq"

    def create_memory_connector(self) -> IMemoryConnector:
        assert self.project_id != "", "Please specify 'project_id'."
        from .gcp import GCPPubSubConnector

        return GCPPubSubConnector(self)
