from abc import ABC, abstractmethod
from typing import Tuple


class IImageBlockConfig(ABC):
    @abstractmethod
    def create_block_tf(self, enable_time_distributed_layer: bool = False):
        raise NotImplementedError()

    @abstractmethod
    def create_block_torch(self, in_shape: Tuple[int, ...], enable_time_distributed_layer: bool = False):
        raise NotImplementedError()


class IAlphaZeroImageBlockConfig(ABC):
    @abstractmethod
    def create_block_tf(self):
        raise NotImplementedError()

    @abstractmethod
    def create_block_torch(self, in_shape: Tuple[int, ...]):
        raise NotImplementedError()


class IMLPBlockConfig(ABC):
    @abstractmethod
    def create_block_tf(self):
        raise NotImplementedError()

    @abstractmethod
    def create_block_torch(self, in_size: int):
        raise NotImplementedError()
