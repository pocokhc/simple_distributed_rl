import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np

from srl.base.define import DoneTypes, InfoType, InvalidActionsType, RLActionType, RLMemoryTypes
from srl.base.render import IRender
from srl.base.rl.config import DummyRLConfig, RLConfig

if TYPE_CHECKING:
    from srl.base.env.env_run import EnvRun
    from srl.base.rl.worker_run import WorkerRun


logger = logging.getLogger(__name__)


# ------------------------------------
# Parameter
# ------------------------------------
class RLParameter(ABC):
    def __init__(self, config: RLConfig):
        self.config = config

    @abstractmethod
    def call_restore(self, data: Any, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def call_backup(self, **kwargs) -> Any:
        raise NotImplementedError()

    def restore(self, data: Any, **kwargs) -> None:
        self.call_restore(data, **kwargs)

    def backup(self, **kwargs) -> Any:
        return self.call_backup(**kwargs)

    def save(self, path: str) -> None:
        logger.debug(f"parameter save: {path}")
        try:
            t0 = time.time()
            with open(path, "wb") as f:
                pickle.dump(self.backup(), f)
            logger.info(f"parameter saved({time.time() - t0:.1f}s): {path}")
        except Exception:
            if os.path.isfile(path):
                os.remove(path)
            raise

    def load(self, path: str) -> None:
        logger.debug(f"parameter load: {path}")
        t0 = time.time()
        with open(path, "rb") as f:
            self.restore(pickle.load(f))
        logger.info(f"parameter loaded({time.time() - t0:.1f}s)")

    def summary(self, **kwargs):
        pass  # NotImplemented


class DummyRLParameter(RLParameter):
    def __init__(self, config: Optional[RLConfig] = None):
        if config is None:
            config = DummyRLConfig()
        super().__init__(config)

    def call_restore(self, data: Any, **kwargs) -> None:
        pass

    def call_backup(self, **kwargs) -> Any:
        return None


# ------------------------------------
# Memory
# ------------------------------------
class _IRLMemoryBase(ABC):
    @property
    @abstractmethod
    def memory_type(self) -> RLMemoryTypes:
        raise NotImplementedError()

    def length(self) -> int:
        return -1


class IRLMemoryWorker(_IRLMemoryBase):
    @abstractmethod
    def add(self, *args) -> None:
        raise NotImplementedError()


class UnusedRLMemoryWorker(IRLMemoryWorker):
    @property
    def memory_type(self) -> RLMemoryTypes:
        return RLMemoryTypes.NONE

    def add(self, *args) -> None:
        raise NotImplementedError("Unused")


class IRLMemoryTrainer(_IRLMemoryBase):
    @abstractmethod
    def is_warmup_needed(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def sample(self, batch_size: int, step: int) -> Any:
        raise NotImplementedError()

    def update(self, memory_update_args: Any) -> None:
        raise NotImplementedError()


class RLMemory(IRLMemoryWorker, IRLMemoryTrainer):
    def __init__(self, config: RLConfig):
        self.config = config

    @abstractmethod
    def call_backup(self, **kwargs) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def call_restore(self, data: Any, **kwargs) -> None:
        raise NotImplementedError()

    def backup(self, compress: bool = False, **kwargs) -> Any:
        dat = self.call_backup(**kwargs)
        if compress:
            import lzma

            dat = pickle.dumps(dat)
            dat = lzma.compress(dat)
            dat = (dat, True)
        return dat

    def restore(self, dat: Any, **kwargs) -> None:
        if isinstance(dat, tuple):
            import lzma

            dat = lzma.decompress(dat[0])
            dat = pickle.loads(dat)
        self.call_restore(dat, **kwargs)

    def save(self, path: str, compress: bool = True, **kwargs) -> None:
        logger.debug(f"memory save (size: {self.length()}): {path}")
        try:
            t0 = time.time()
            dat = self.call_backup(**kwargs)
            if compress:
                import lzma

                dat = pickle.dumps(dat)
                with lzma.open(path, "w") as f:
                    f.write(dat)
            else:
                with open(path, "wb") as f:
                    pickle.dump(dat, f)
            logger.info(f"memory saved (size: {self.length()}, time: {time.time() - t0:.1f}s): {path}")
        except Exception:
            if os.path.isfile(path):
                os.remove(path)
            raise

    def load(self, path: str, **kwargs) -> None:
        import binascii

        logger.debug(f"memory load: {path}")
        t0 = time.time()
        # LZMA
        with open(path, "rb") as f:
            compress = binascii.hexlify(f.read(6)) == b"fd377a585a00"
        if compress:
            import lzma

            with lzma.open(path) as f:
                dat = f.read()
            dat = pickle.loads(dat)
        else:
            with open(path, "rb") as f:
                dat = pickle.load(f)
        self.call_restore(dat, **kwargs)
        logger.info(f"memory loaded (size: {self.length()}, time: {time.time() - t0:.1f}s): {path}")


class DummyRLMemory(RLMemory):
    def __init__(self, config: Optional[RLConfig] = None):
        if config is None:
            config = DummyRLConfig()
        super().__init__(config)

    @property
    def memory_type(self) -> RLMemoryTypes:
        return RLMemoryTypes.NONE

    def call_backup(self, **kwargs) -> Any:
        return None

    def call_restore(self, data: Any, **kwargs) -> None:
        pass

    def add(self, *args) -> None:
        pass

    def is_warmup_needed(self) -> bool:
        return False

    def sample(self, batch_size: int, step: int) -> Any:
        return None


# ------------------------------------
# Trainer
# ------------------------------------
class RLTrainer(ABC):
    def __init__(
        self,
        config: RLConfig,
        parameter: RLParameter,
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

        # abstract value
        self.train_count: int = 0
        self.train_info: InfoType = {}

    def get_train_count(self) -> int:
        return self.train_count

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError()

    # abstract
    def train_start(self) -> None:
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


# ------------------------------------
# Worker
# ------------------------------------
class RLWorker(ABC, IRender):
    def __init__(
        self, config: RLConfig, parameter: Optional[RLParameter] = None, memory: Optional[IRLMemoryWorker] = None
    ) -> None:
        self.config = config
        self.parameter = parameter
        self.memory = UnusedRLMemoryWorker() if memory is None else memory

    def _set_worker_run(self, worker: "WorkerRun"):
        """WorkerRunの初期化で呼ばれる"""
        self.__worker_run = worker

    # ------------------------------
    # implement
    # ------------------------------
    def on_reset(self, worker: "WorkerRun") -> InfoType:
        return {}

    @abstractmethod
    def policy(self, worker: "WorkerRun") -> Tuple[RLActionType, InfoType]:
        raise NotImplementedError()

    def on_step(self, worker: "WorkerRun") -> InfoType:
        return {}

    def on_start(self, worker: "WorkerRun") -> None:
        pass

    def on_end(self, worker: "WorkerRun") -> None:
        pass

    # ------------------------------
    # IRender
    # ------------------------------
    def render_terminal(self, worker: "WorkerRun", **kwargs) -> None:
        pass

    def render_rgb_array(self, worker: "WorkerRun", **kwargs) -> Optional[np.ndarray]:
        return None

    # ------------------------------------
    # instance
    # ------------------------------------
    @property
    def worker(self) -> "WorkerRun":
        return self.__worker_run

    @property
    def env(self) -> "EnvRun":
        return self.__worker_run._env

    def terminated(self) -> None:
        self.__worker_run._env._done = DoneTypes.TRUNCATED
        self.__worker_run._env._done_reason = "rl"

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
        return self.__worker_run._env.max_episode_steps

    @property
    def player_num(self) -> int:
        return self.__worker_run._env.player_num

    @property
    def step(self) -> int:
        return self.__worker_run._env.step_num
