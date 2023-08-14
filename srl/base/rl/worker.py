from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from srl.base.define import InfoType, InvalidActionsType, RLActionType
from srl.base.render import IRender
from srl.base.rl.base import RLParameter, RLRemoteMemory
from srl.base.rl.config import DummyConfig, RLConfig

if TYPE_CHECKING:
    from srl.base.rl.worker_run import WorkerRun


class WorkerBase(ABC, IRender):
    def __init__(
        self,
        config: Optional[RLConfig] = None,
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
    ) -> None:
        if config is None:
            config = DummyConfig()
        self.config: RLConfig = config
        self.parameter = parameter
        self.remote_memory = remote_memory

    def _set_worker_run(self, worker: "WorkerRun"):
        """WorkerRunの初期化で呼ばれる"""
        self.__worker_run = worker

    # ------------------------------
    # implement
    # ------------------------------
    @abstractmethod
    def on_reset(self, worker: "WorkerRun") -> InfoType:
        raise NotImplementedError()

    @abstractmethod
    def policy(self, worker: "WorkerRun") -> Tuple[RLActionType, InfoType]:
        raise NotImplementedError()

    @abstractmethod
    def on_step(self, worker: "WorkerRun") -> InfoType:
        raise NotImplementedError()

    # ------------------------------
    # implement(option)
    # ------------------------------
    def render_terminal(self, worker: "WorkerRun", **kwargs) -> None:
        pass

    def render_rgb_array(self, worker: "WorkerRun", **kwargs) -> Optional[np.ndarray]:
        return None

    # ------------------------------------
    # instance
    # ------------------------------------
    @property
    def worker_run(self) -> "WorkerRun":
        return self.__worker_run

    # ------------------------------------
    # worker info (shortcut properties)
    # ------------------------------------
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

    def get_invalid_actions(self) -> InvalidActionsType:
        return self.__worker_run.get_invalid_actions()

    def sample_action(self) -> RLActionType:
        return self.__worker_run.sample_action()

    # ------------------------------------
    # env info (shortcut properties)
    # ------------------------------------
    @property
    def max_episode_steps(self) -> int:
        return self.__worker_run.env.max_episode_steps

    @property
    def player_num(self) -> int:
        return self.__worker_run.env.player_num

    @property
    def step(self) -> int:
        return self.__worker_run.env.step_num
