import logging
import os
import pickle
import time
from abc import ABC
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, Tuple, TypeVar, cast

from srl.base.rl.config import DummyRLConfig, TRLConfig
from srl.utils.common import load_file, save_file

logger = logging.getLogger(__name__)

TRLMemory = TypeVar("TRLMemory", bound="RLMemory", covariant=True)


class WorkerFunc(Protocol):
    def __call__(self, *args, serialized: bool = False) -> None: ...

    @property
    def __name__(self) -> str: ...


class WorkerSerializeFunc(Protocol):
    def __call__(self, *args, **kwargs) -> Any: ...


class TrainerRecvFunc(Protocol):
    def __call__(self) -> Optional[Any]: ...

    @property
    def __name__(self) -> str: ...


class TrainerSendFunc(Protocol):
    def __call__(self, *args, **kwargs) -> None: ...

    @property
    def __name__(self) -> str: ...


class RLMemory(ABC, Generic[TRLConfig]):
    def __init__(self, config: Optional[TRLConfig]):
        self.config: TRLConfig = cast(TRLConfig, DummyRLConfig()) if config is None else config
        self.__worker_funcs: Dict[str, Tuple[WorkerFunc, WorkerSerializeFunc]] = {}
        self.__trainer_recv_funcs: List[TrainerRecvFunc] = []
        self.__trainer_send_funcs: Dict[str, TrainerSendFunc] = {}
        self.setup()

    def setup(self) -> None:
        pass  # NotImplemented

    def register_worker_func(self, func: Callable, serialize_func: WorkerSerializeFunc):
        """
        serialize_funcは引数展開用にtuple形式に変換する。
        その条件にtupleかどうかを見ているので戻り値1つの場合にtupleは使用できない。
        """
        if func.__name__ in self.__worker_funcs:
            logger.warning(f"'{func.__name__}' is already registered. It has been overwritten.")

        self.__worker_funcs[func.__name__] = (func, serialize_func)

    def get_worker_funcs(self):
        return self.__worker_funcs

    def register_trainer_recv_func(self, func: TrainerRecvFunc):
        self.__trainer_recv_funcs.append(func)

    def get_trainer_recv_funcs(self):
        return self.__trainer_recv_funcs

    def register_trainer_send_func(self, func: TrainerSendFunc):
        if func.__name__ in self.__trainer_send_funcs:
            logger.warning(f"'{func.__name__}' is already registered. It has been overwritten.")
        self.__trainer_send_funcs[func.__name__] = func

    def get_trainer_send_funcs(self):
        return self.__trainer_send_funcs

    def length(self) -> int:
        return -1

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
        logger.debug(f"memory save (len: {self.length()}): {path}")
        t0 = time.time()
        dat = self.call_backup(**kwargs)
        save_file(path, dat, compress)
        file_size = os.path.getsize(path)
        logger.info(f"memory saved (len: {self.length()}, {file_size} bytes(compress {compress}), time: {time.time() - t0:.1f}s): {os.path.basename(path)}")

    def load(self, path: str, **kwargs) -> None:
        logger.debug(f"memory load: {path}")
        t0 = time.time()
        dat = load_file(path)
        self.call_restore(dat, **kwargs)
        logger.info(f"memory loaded (size: {self.length()}, time: {time.time() - t0:.1f}s): {os.path.basename(path)}")


class DummyRLMemory(RLMemory):
    def call_backup(self, **kwargs) -> Any:
        return None

    def call_restore(self, data: Any, **kwargs) -> None:
        pass
