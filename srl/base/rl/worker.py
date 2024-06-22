import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, List, Optional, cast

import numpy as np

from srl.base.define import (
    DoneTypes,
    RLActionType,
    RLObservationType,
    TActSpace,
    TActType,
    TConfig,
    TObsSpace,
    TObsType,
    TParameter,
)
from srl.base.info import Info
from srl.base.render import IRender
from srl.base.rl.config import RLConfig
from srl.base.rl.memory import DummyRLMemoryWorker, IRLMemoryWorker
from srl.base.rl.parameter import DummyRLParameter
from srl.base.spaces.space import SpaceBase

if TYPE_CHECKING:
    from srl.base.context import RunContext
    from srl.base.env.env_run import EnvRun
    from srl.base.rl.worker_run import WorkerRun


logger = logging.getLogger(__name__)


class RLWorkerGeneric(
    ABC,
    IRender,
    Generic[TConfig, TParameter, TActSpace, TActType, TObsSpace, TObsType],
):
    def __init__(
        self,
        config: TConfig,
        parameter: Optional[TParameter] = None,
        memory: Optional[IRLMemoryWorker] = None,
    ) -> None:
        self.config = config
        self.parameter: TParameter = cast(
            TParameter,
            DummyRLParameter(cast(RLConfig, config)) if parameter is None else parameter,
        )
        self.memory: IRLMemoryWorker = cast(IRLMemoryWorker, DummyRLMemoryWorker() if memory is None else memory)

        # abstract value
        self.info = Info()

    def _set_worker_run(self, worker: "WorkerRun[TActSpace, TActType, TObsSpace, TObsType]"):
        """WorkerRunの初期化で呼ばれる"""
        self.__worker_run = worker

    # --- implement
    def on_start(self, worker: "WorkerRun[TActSpace, TActType, TObsSpace, TObsType]", context: "RunContext") -> None:
        pass

    def on_reset(self, worker: "WorkerRun[TActSpace, TActType, TObsSpace, TObsType]") -> None:
        pass

    @abstractmethod
    def policy(self, worker: "WorkerRun[TActSpace, TActType, TObsSpace, TObsType]") -> TActType:
        raise NotImplementedError()

    def on_step(self, worker: "WorkerRun[TActSpace, TActType, TObsSpace, TObsType]") -> None:
        pass

    def on_end(self, worker: "WorkerRun[TActSpace, TActType, TObsSpace, TObsType]") -> None:
        pass

    # --- IRender
    def render_terminal(self, worker: "WorkerRun[TActSpace, TActType, TObsSpace, TObsType]", **kwargs) -> None:
        pass

    def render_rgb_array(
        self, worker: "WorkerRun[TActSpace, TActType, TObsSpace, TObsType]", **kwargs
    ) -> Optional[np.ndarray]:
        return None

    # --- instance
    @property
    def worker(self) -> "WorkerRun[TActSpace, TActType, TObsSpace, TObsType]":
        return self.__worker_run

    @property
    def env(self) -> "EnvRun":
        return self.__worker_run._env

    def terminated(self) -> None:
        self.__worker_run._env._done = DoneTypes.TRUNCATED
        self.__worker_run._env._done_reason = "rl"

    # --- worker info (shortcut properties)
    @property
    def training(self) -> bool:
        return self.__worker_run.training

    @property
    def distributed(self) -> bool:
        return self.__worker_run.distributed

    @property
    def rendering(self) -> bool:
        return self.__worker_run.rendering

    @property
    def player_index(self) -> int:
        return self.__worker_run.player_index

    @property
    def total_step(self) -> int:
        return self.__worker_run.total_step

    def get_invalid_actions(self) -> List[TActType]:
        return self.__worker_run.get_invalid_actions()

    def sample_action(self) -> TActType:
        return self.__worker_run.sample_action()

    # --- env info (shortcut properties)
    @property
    def max_episode_steps(self) -> int:
        return self.__worker_run._env.max_episode_steps

    @property
    def player_num(self) -> int:
        return self.__worker_run._env.player_num

    @property
    def step(self) -> int:
        return self.__worker_run._env.step_num


class RLWorker(
    Generic[TConfig, TParameter],
    RLWorkerGeneric[
        TConfig,
        TParameter,
        SpaceBase,
        RLActionType,
        SpaceBase,
        RLObservationType,
    ],
):
    pass


class DummyRLWorker(RLWorker):
    def policy(self, worker: "WorkerRun") -> RLActionType:
        return worker.sample_action()
