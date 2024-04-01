import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, List, Optional, Tuple, TypeVar, cast

import numpy as np

from srl.base.define import DoneTypes, InfoType, RLActionType, RLInvalidActionType
from srl.base.render import IRender
from srl.base.rl.config import RLConfig
from srl.base.rl.memory import DummyRLMemoryWorker, IRLMemoryWorker
from srl.base.rl.parameter import DummyRLParameter

if TYPE_CHECKING:
    from srl.base.context import RunContext
    from srl.base.env.env_run import EnvRun
    from srl.base.rl.worker_run import WorkerRun


logger = logging.getLogger(__name__)

_TConfig = TypeVar("_TConfig")
_TParameter = TypeVar("_TParameter")


class RLWorker(ABC, IRender, Generic[_TConfig, _TParameter]):
    def __init__(
        self,
        config: _TConfig,
        parameter: Optional[_TParameter] = None,
        memory: Optional[IRLMemoryWorker] = None,
    ) -> None:
        self.config = config
        self.parameter: _TParameter = cast(
            _TParameter,
            DummyRLParameter(cast(RLConfig, config)) if parameter is None else parameter,
        )
        self.memory: IRLMemoryWorker = cast(IRLMemoryWorker, DummyRLMemoryWorker() if memory is None else memory)

    def _set_worker_run(self, worker: "WorkerRun"):
        """WorkerRunの初期化で呼ばれる"""
        self.__worker_run = worker

    # --- implement
    def on_start(self, worker: "WorkerRun", context: "RunContext") -> None:
        pass

    def on_reset(self, worker: "WorkerRun") -> InfoType:
        return {}

    @abstractmethod
    def policy(self, worker: "WorkerRun") -> Tuple[RLActionType, InfoType]:
        raise NotImplementedError()

    def on_step(self, worker: "WorkerRun") -> InfoType:
        return {}

    def on_end(self, worker: "WorkerRun") -> None:
        pass

    # --- IRender
    def render_terminal(self, worker: "WorkerRun", **kwargs) -> None:
        pass

    def render_rgb_array(self, worker: "WorkerRun", **kwargs) -> Optional[np.ndarray]:
        return None

    # --- instance
    @property
    def worker(self) -> "WorkerRun":
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

    def get_invalid_actions(self) -> List[RLInvalidActionType]:
        return self.__worker_run.get_invalid_actions()

    def sample_action(self) -> RLActionType:
        return self.__worker_run.sample_action()

    @property
    def observation_space(self):
        return cast(RLConfig, self.config).observation_space

    @property
    def action_space(self):
        return cast(RLConfig, self.config).action_space

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


class DummyRLWorker(RLWorker):
    def policy(self, worker: "WorkerRun") -> Tuple[RLActionType, dict]:
        return worker.sample_action(), {}
