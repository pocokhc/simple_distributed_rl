from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class IServerParameters(ABC):
    keepalive_interval: int = 10

    @abstractmethod
    def create_memory_connector(self) -> "IMemoryConnector":
        raise NotImplementedError()


class IMemoryConnector(ABC):
    @abstractmethod
    def ping(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def memory_add(self, dat: Any) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def memory_recv(self) -> Optional[Any]:
        raise NotImplementedError()

    @abstractmethod
    def memory_size(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def memory_purge(self) -> None:
        raise NotImplementedError()
