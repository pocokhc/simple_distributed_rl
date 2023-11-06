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
    def memory_setup(self, task_id: str):
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
    def memory_delete_if_exist(self, task_id: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def memory_exist(self, task_id: str) -> bool:
        raise NotImplementedError()
