import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, cast

from srl.base.define import RLMemoryTypes, TConfig
from srl.base.rl.config import DummyRLConfig, RLConfig

logger = logging.getLogger(__name__)


class IRLMemoryWorker(ABC):
    @property
    def memory_type(self) -> RLMemoryTypes:
        return RLMemoryTypes.UNKNOWN

    def length(self) -> int:
        return -1

    @abstractmethod
    def add(self, *args) -> None:
        raise NotImplementedError()

    def serialize_add_args(self, *args) -> Any:
        return pickle.dumps(args)

    def deserialize_add_args(self, raw: Any) -> Any:
        return pickle.loads(raw)


class RLMemory(IRLMemoryWorker, Generic[TConfig]):
    def __init__(self, config: Optional[TConfig]):
        self.config: TConfig = cast(TConfig, DummyRLConfig() if config is None else config)

    def call_backup(self, **kwargs) -> Any:
        raise NotImplementedError()

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


class DummyRLMemoryWorker(IRLMemoryWorker):
    def add(self, *args) -> None:
        raise NotImplementedError()


class DummyRLMemory(RLMemory):
    def call_backup(self, **kwargs) -> Any:
        return None

    def call_restore(self, data: Any, **kwargs) -> None:
        pass

    def add(self, *args) -> None:
        pass
