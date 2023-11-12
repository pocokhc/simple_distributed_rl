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
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Serverへの接続無し。例外は出さない"""
        raise NotImplementedError()

    @abstractmethod
    def ping(self) -> bool:
        """Serverへ接続状況を問い合わせる。例外は出さない"""
        raise NotImplementedError()

    @abstractmethod
    def memory_add(self, dat: Any) -> None:
        """例外を出す"""
        raise NotImplementedError()

    @abstractmethod
    def memory_recv(self) -> Optional[Any]:
        """例外を出す"""
        raise NotImplementedError()

    @abstractmethod
    def memory_size(self) -> int:
        """例外は出さない、失敗時は-1"""
        raise NotImplementedError()

    @abstractmethod
    def memory_purge(self) -> None:
        """例外は出さない"""
        raise NotImplementedError()
