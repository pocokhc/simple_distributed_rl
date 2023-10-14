import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from typing import Any

from srl.base.define import InfoType
from srl.base.rl.config import RLConfig

logger = logging.getLogger(__name__)


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


class RLRemoteMemory(ABC):
    def __init__(self, config: RLConfig):
        self.config = config

    @abstractmethod
    def length(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def call_restore(self, data: Any, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def call_backup(self, **kwargs) -> Any:
        raise NotImplementedError()

    def restore(self, dat: Any, **kwargs) -> None:
        if isinstance(dat, tuple):
            import lzma

            dat = lzma.decompress(dat[0])
            dat = pickle.loads(dat)
        self.call_restore(dat, **kwargs)

    def backup(self, compress: bool = False, **kwargs) -> Any:
        dat = self.call_backup(**kwargs)
        if compress:
            import lzma

            dat = pickle.dumps(dat)
            dat = lzma.compress(dat)
            dat = (dat, True)
        return dat

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


class RLTrainer(ABC):
    def __init__(self, config: RLConfig, parameter: RLParameter, memory: RLMemory):
        self.config = config
        self.parameter = parameter
        self.memory = memory

        self.batch_size: int = getattr(self.config, "batch_size", -1)

        self.train_count: int = 0
        self.train_info: InfoType = {}

    def get_train_count(self) -> int:
        return self.train_count

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        memory_sample_return = self.memory.sample(self.batch_size, self.train_count)
        self.train_on_batchs(memory_sample_return)

    def memory_update(self, memory_update_args):
        self.memory.update(memory_update_args)

    @abstractmethod
    def train_on_batchs(self, memory_sample_return) -> None:
        raise NotImplementedError()
