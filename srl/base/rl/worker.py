import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Optional

import numpy as np

from srl.base.define import DoneTypes, RLActionType, RLObservationType
from srl.base.info import Info
from srl.base.render import IRender
from srl.base.rl.config import TRLConfig
from srl.base.rl.memory import DummyRLMemory, TRLMemory
from srl.base.rl.parameter import DummyRLParameter, TRLParameter
from srl.base.spaces.space import SpaceBase, TActSpace, TActType, TObsSpace, TObsType

if TYPE_CHECKING:
    from srl.base.context import RunContext
    from srl.base.env.env_run import EnvRun
    from srl.base.rl.worker_run import WorkerRun


logger = logging.getLogger(__name__)


class RLWorkerGeneric(
    IRender,
    Generic[TRLConfig, TRLParameter, TRLMemory, TActSpace, TActType, TObsSpace, TObsType],
    ABC,
):
    def __init__(
        self,
        config: TRLConfig,
        parameter: Optional[TRLParameter] = None,
        memory: Optional[TRLMemory] = None,
    ) -> None:
        self.config = config
        self.parameter: TRLParameter = DummyRLParameter(config) if parameter is None else parameter
        self.memory: TRLMemory = DummyRLMemory(config) if memory is None else memory

        # abstract value
        self.info = Info()

    def _set_worker_run(self, worker: "WorkerRun[TActSpace, TActType, TObsSpace, TObsType]"):
        """WorkerRunの初期化で呼ばれる"""
        self.__worker_run = worker

    # --- implement
    def on_setup(self, worker: "WorkerRun[TActSpace, TActType, TObsSpace, TObsType]", context: "RunContext") -> None:
        pass

    def on_teardown(self, worker: "WorkerRun[TActSpace, TActType, TObsSpace, TObsType]") -> None:
        pass

    def on_reset(self, worker: "WorkerRun[TActSpace, TActType, TObsSpace, TObsType]") -> None:
        pass

    @abstractmethod
    def policy(self, worker: "WorkerRun[TActSpace, TActType, TObsSpace, TObsType]") -> TActType:
        raise NotImplementedError()

    def on_step(self, worker: "WorkerRun[TActSpace, TActType, TObsSpace, TObsType]") -> None:
        pass

    # --- IRender
    def render_terminal(self, worker: "WorkerRun[TActSpace, TActType, TObsSpace, TObsType]", **kwargs) -> None:
        pass

    def render_rgb_array(self, worker: "WorkerRun[TActSpace, TActType, TObsSpace, TObsType]", **kwargs) -> Optional[np.ndarray]:
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
        self.__worker_run._env.env.done_reason = "rl"

    # --- worker info (shortcut properties)
    @property
    def context(self) -> "RunContext":
        return self.__worker_run._context

    @property
    def distributed(self) -> bool:
        return self.__worker_run._context.distributed

    @property
    def training(self) -> bool:
        return self.__worker_run._context.training

    @property
    def train_only(self) -> bool:
        return self.__worker_run._context.train_only

    @property
    def rollout(self) -> bool:
        return self.__worker_run._context.rollout

    @property
    def rendering(self) -> bool:
        return self.__worker_run._context.rendering

    @property
    def player_index(self) -> int:
        return self.__worker_run.player_index

    @property
    def total_step(self) -> int:
        return self.__worker_run.total_step

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

    # --- utils
    def sample_action(self) -> TActType:
        return self.__worker_run.sample_action()


class RLWorker(
    Generic[TRLConfig, TRLParameter, TRLMemory],
    RLWorkerGeneric[
        TRLConfig,
        TRLParameter,
        TRLMemory,
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
