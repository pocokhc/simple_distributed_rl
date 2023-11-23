from abc import ABC, abstractmethod
from dataclasses import dataclass
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
class IParameterWriter(IServerConnector):
    @abstractmethod
    def parameter_update(self, parameter: Any):
        """例外を出す"""
        raise NotImplementedError()


class IParameterReader(IServerConnector):
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
    def memory_add(self, dat: Any) -> None:
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
