import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

from srl.base.define import InfoType, RLActionType, RLBaseTypes, RLInvalidActionType, RLObservationType
from srl.base.rl.config import RLConfig
from srl.base.rl.worker import RLWorker
from srl.base.rl.worker_run import WorkerRun

logger = logging.getLogger(__name__)


@dataclass
class DiscreteActionConfig(RLConfig):
    @property
    def base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.DISCRETE


class DiscreteActionWorker(RLWorker):
    @abstractmethod
    def call_on_reset(
        self,
        state: RLObservationType,
        invalid_actions: List[RLInvalidActionType],
    ) -> InfoType:
        raise NotImplementedError()

    @abstractmethod
    def call_policy(
        self,
        state: RLObservationType,
        invalid_actions: List[RLInvalidActionType],
    ) -> Tuple[int, InfoType]:
        raise NotImplementedError()

    @abstractmethod
    def call_on_step(
        self,
        next_state: RLObservationType,
        reward: float,
        done: bool,
        next_invalid_actions: List[RLInvalidActionType],
    ) -> InfoType:
        raise NotImplementedError()

    # --------------------------------------
    def on_reset(self, worker: WorkerRun) -> InfoType:
        return self.call_on_reset(worker.state, worker.get_invalid_actions())

    def policy(self, worker: WorkerRun) -> Tuple[RLActionType, InfoType]:
        return self.call_policy(worker.state, worker.get_invalid_actions())

    def on_step(self, worker: WorkerRun) -> InfoType:
        return self.call_on_step(
            worker.state,
            worker.reward,
            worker.done,
            worker.get_invalid_actions(),
        )
