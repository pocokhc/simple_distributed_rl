import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, Optional

from srl.base.context import RunContext
from srl.base.define import InfoType, TConfig, TParameter
from srl.base.rl.memory import IRLMemoryTrainer

logger = logging.getLogger(__name__)

FLAG_1STEP = "1STEP"


class RLTrainer(ABC, Generic[TConfig, TParameter]):
    def __init__(
        self,
        config: TConfig,
        parameter: TParameter,
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
        self.info: InfoType = {}

        # abstract value
        self.train_count: int = 0

    def get_train_count(self) -> int:
        return self.train_count

    # --- 3step train
    def train_setup(self) -> Optional[Any]:
        # return setup_data
        return FLAG_1STEP

    @abstractmethod
    def train(self, setup_data: Any) -> Any:
        # return run_data
        raise NotImplementedError()

    def train_teardown(self, run_data: Any) -> None:
        pass

    # --- funcs
    # abstract
    def train_start(self, context: RunContext) -> None:
        pass

    # abstract
    def train_end(self) -> None:
        pass

    # --- properties
    @property
    def distributed(self) -> bool:
        return self.__distributed

    @property
    def train_only(self) -> bool:
        return self.__train_only

    # ----------------
    def core_train(self) -> bool:
        setup_data = self.train_setup()
        if setup_data is None:
            return False
        _prev_train = self.train_count
        if setup_data == FLAG_1STEP:
            run_data = self.train()  # 互換用
        else:
            run_data = self.train(setup_data)
        self.train_teardown(run_data)
        return self.train_count > _prev_train


class DummyRLTrainer(RLTrainer):
    def train(self) -> None:
        self.train_count += 1
