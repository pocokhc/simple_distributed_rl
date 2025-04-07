import logging
from abc import ABC
from typing import Generic

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

    @property
    def context(self) -> RunContext:
        return self.__context

    @property
    def distributed(self) -> bool:
        return self.__context.distributed

    @property
    def train_only(self) -> bool:
        return self.__context.train_only

    def setup(self, context: RunContext) -> None:
        self.__context = context
        self.on_setup()

    def teardown(self) -> None:
        self.on_teardown()

    # --------------------------------
    # implement
    # --------------------------------
    def on_setup(self) -> None:
        pass

    def on_teardown(self) -> None:
        pass

    def train(self) -> None:
        raise NotImplementedError()


class DummyRLTrainer(RLTrainer):
    def train(self) -> None:
        self.train_count += 1
