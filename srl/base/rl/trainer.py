import logging
from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from srl.base.define import InfoType
from srl.base.rl.memory import IRLMemoryTrainer

logger = logging.getLogger(__name__)

_TConfig = TypeVar("_TConfig")
_TParameter = TypeVar("_TParameter")


class RLTrainer(ABC, Generic[_TConfig, _TParameter]):
    def __init__(
        self,
        config: _TConfig,
        parameter: _TParameter,
        memory: IRLMemoryTrainer,
        distributed: bool = False,
        train_only: bool = False,
    ):
        self.config = config
        self.parameter = parameter
        self.memory = memory
        self.__distributed = distributed
        self.__train_only = train_only

        self.batch_size: int = getattr(self.config, "batch_size", 1)
        self.info: Optional[InfoType] = None

        # abstract value
        self.train_count: int = 0

    def get_train_count(self) -> int:
        return self.train_count

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError()

    # abstract
    def train_start(self) -> None:
        pass

    # abstract
    def train_end(self) -> None:
        pass

    # abstract
    def create_info(self) -> InfoType:
        return {}

    def get_info(self) -> InfoType:
        if self.info is None:
            self.info = self.create_info()
        return self.info

    # --- properties
    @property
    def distributed(self) -> bool:
        return self.__distributed

    @property
    def train_only(self) -> bool:
        return self.__train_only

    # ----------------
    def core_train(self) -> bool:
        self.info = None
        _prev_train = self.train_count
        self.train()
        return self.train_count > _prev_train


class DummyRLTrainer(RLTrainer):
    def train(self) -> None:
        self.train_count += 1
