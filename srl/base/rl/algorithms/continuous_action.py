import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from srl.base.define import InfoType, RLActionType, RLBaseTypes
from srl.base.rl.config import RLConfig
from srl.base.rl.worker import RLWorker
from srl.base.rl.worker_run import WorkerRun

logger = logging.getLogger(__name__)


@dataclass
class ContinuousActionConfig(RLConfig):
    @property
    def base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.CONTINUOUS


class ContinuousActionWorker(RLWorker):
    @abstractmethod
    def call_on_reset(self, state: np.ndarray) -> InfoType:
        raise NotImplementedError()

    @abstractmethod
    def call_policy(self, state: np.ndarray) -> Tuple[List[float], InfoType]:
        raise NotImplementedError()

    @abstractmethod
    def call_on_step(
        self,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ) -> InfoType:
        raise NotImplementedError()

    # ----------------------------------
    def on_reset(self, worker: WorkerRun) -> InfoType:
        return self.call_on_reset(worker.state)

    def policy(self, worker: WorkerRun) -> Tuple[RLActionType, InfoType]:
        return self.call_policy(worker.state)

    def on_step(self, worker: WorkerRun) -> InfoType:
        return self.call_on_step(worker.state, worker.reward, worker.done)
