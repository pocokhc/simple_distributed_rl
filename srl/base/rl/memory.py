import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar, cast

from srl.base.define import RLMemoryTypes
from srl.base.rl.config import DummyRLConfig, RLConfig

logger = logging.getLogger(__name__)

_TConfig = TypeVar("_TConfig")


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


class DummyRLMemoryWorker(IRLMemoryWorker):
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
    def sample(self, *args) -> Any:
        raise NotImplementedError()

    def update(self, *args) -> None:
        raise NotImplementedError()


class RLMemory(IRLMemoryWorker, IRLMemoryTrainer, Generic[_TConfig]):
    def __init__(self, config: Optional[_TConfig]):
        self.config: _TConfig = cast(_TConfig, DummyRLConfig() if config is None else config)
        self.compress = cast(RLConfig, self.config).memory_compress

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

    def conditional_compress(self, dat: Any) -> bytes:
        if not self.compress:
            return dat
        import zlib

        return zlib.compress(pickle.dumps(dat))

    def conditional_decompress(self, raw: bytes) -> Any:
        if not self.compress:
            return raw
        import zlib

        return pickle.loads(zlib.decompress(raw))


class DummyRLMemory(RLMemory):
    def __init__(self, config: Optional[RLConfig] = None):
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
