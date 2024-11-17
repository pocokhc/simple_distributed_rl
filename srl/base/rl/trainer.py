import logging
from abc import ABC
from typing import Any, Generic, Optional

from srl.base.context import RunContext
from srl.base.info import Info
from srl.base.rl.config import TRLConfig
from srl.base.rl.memory import TRLMemory
from srl.base.rl.parameter import TRLParameter

logger = logging.getLogger(__name__)


class RLTrainer(ABC, Generic[TRLConfig, TRLParameter, TRLMemory]):
    def __init__(self, config: TRLConfig, parameter: TRLParameter, memory: TRLMemory):
        self.config = config
        self.parameter = parameter
        self.memory = memory
        self.__context = RunContext()

        # abstract value
        self.train_count: int = 0
        self.info = Info()

    def get_train_count(self) -> int:
        return self.train_count

    def on_start(self, context: RunContext) -> None:
        self.__context = context

    def on_end(self) -> None:
        pass

    @property
    def distributed(self) -> bool:
        return self.__context.distributed

    @property
    def train_only(self) -> bool:
        return self.__context.train_only

    # --- 1step train
    def train(self) -> None:
        raise NotImplementedError()

    # --- 3step train
    def implement_thread_train(self) -> bool:
        #: 仮実装なので、これがTrueの場合のみ有効
        return False

    def thread_train_setup(self) -> Optional[Any]:
        # return setup_data
        return None

    def thread_train(self, setup_data: Any) -> Any:
        # return train_data
        raise NotImplementedError()

    def thread_train_teardown(self, train_data: Any) -> None:
        pass


class DummyRLTrainer(RLTrainer):
    def train(self) -> None:
        self.train_count += 1

    def thread_train(self, setup_data: Any) -> Any:
        self.train_count += 1
        return None
