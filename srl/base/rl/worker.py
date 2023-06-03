import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple, cast

import numpy as np

from srl.base.define import (
    EnvActionType,
    EnvObservationType,
    InfoType,
    InvalidActionsType,
    InvalidActionType,
    RLActionType,
    RLObservationType,
    RLTypes,
)
from srl.base.env.env_run import EnvRun
from srl.base.render import IRender
from srl.base.rl.base import RLParameter, RLRemoteMemory
from srl.base.rl.config import RLConfig

if TYPE_CHECKING:
    from srl.base.rl.worker_run import WorkerRun

logger = logging.getLogger(__name__)


class WorkerBase(ABC, IRender):
    def __init__(self, training: bool, distributed: bool):
        self.__training = training
        self.__distributed = distributed

    # ------------------------------
    # implement
    # ------------------------------
    @property
    @abstractmethod
    def player_index(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def on_reset(self, env: EnvRun, worker: "WorkerRun") -> InfoType:
        raise NotImplementedError()

    @abstractmethod
    def policy(self, env: EnvRun, worker: "WorkerRun") -> Tuple[EnvActionType, InfoType]:
        raise NotImplementedError()

    @abstractmethod
    def on_step(self, env: EnvRun, worker: "WorkerRun") -> InfoType:
        raise NotImplementedError()

    # ------------------------------
    # implement(option)
    # ------------------------------
    def render_terminal(self, env: EnvRun, worker: "WorkerRun", **kwargs) -> None:
        pass

    def render_rgb_array(self, env: EnvRun, worker: "WorkerRun", **kwargs) -> Optional[np.ndarray]:
        return None

    # ------------------------------------
    # run properties
    # ------------------------------------
    @property
    def training(self) -> bool:
        return self.__training

    @property
    def distributed(self) -> bool:
        return self.__distributed


class RuleBaseWorker(WorkerBase):
    def call_on_reset(self, env: EnvRun, worker: "WorkerRun") -> InfoType:
        return {}

    @abstractmethod
    def call_policy(self, env: EnvRun, worker: "WorkerRun") -> Tuple[EnvActionType, InfoType]:
        raise NotImplementedError()

    def call_on_step(self, env: EnvRun, worker: "WorkerRun") -> InfoType:
        return {}  # do nothing

    @property
    def player_index(self) -> int:
        return self._player_index

    def on_reset(self, env: EnvRun, worker: "WorkerRun") -> InfoType:
        self._player_index = worker.player_index
        return self.call_on_reset(env, worker)

    def policy(self, env: EnvRun, worker: "WorkerRun") -> Tuple[EnvActionType, InfoType]:
        return self.call_policy(env, worker)

    def on_step(self, env: EnvRun, worker: "WorkerRun") -> InfoType:
        return self.call_on_step(env, worker)


class ExtendWorker(WorkerBase):
    def __init__(
        self,
        rl_worker: "WorkerRun",
        training: bool,
        distributed: bool,
    ):
        super().__init__(training, distributed)
        self.rl_worker = rl_worker
        self.worker = self.rl_worker.worker

    @abstractmethod
    def call_on_reset(self, env: EnvRun, worker: "WorkerRun") -> InfoType:
        raise NotImplementedError()

    @abstractmethod
    def call_policy(self, env: EnvRun, worker: "WorkerRun") -> Tuple[EnvActionType, InfoType]:
        raise NotImplementedError()

    def call_on_step(self, env: EnvRun, worker: "WorkerRun") -> InfoType:
        return {}  # do nothing

    @property
    def player_index(self) -> int:
        return self._player_index

    def on_reset(self, env: EnvRun, worker: "WorkerRun") -> InfoType:
        self._player_index = worker.player_index
        self.rl_worker.on_reset(env, worker.player_index)
        return self.call_on_reset(env, worker)

    def policy(self, env: EnvRun, worker: "WorkerRun") -> Tuple[EnvActionType, InfoType]:
        return self.call_policy(env, worker)

    def on_step(self, env: EnvRun, worker: "WorkerRun") -> InfoType:
        self.rl_worker.on_step(env)
        return self.call_on_step(env, worker)
