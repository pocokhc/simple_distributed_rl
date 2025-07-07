import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar, cast

from srl.base.rl.config import DummyRLConfig, RLConfig, TRLConfig
from srl.utils.common import load_file, save_file

logger = logging.getLogger(__name__)

TRLParameter = TypeVar("TRLParameter", bound="RLParameter", covariant=True)


class RLParameter(ABC, Generic[TRLConfig]):
    def __init__(self, config: Optional[TRLConfig]):
        self.config: TRLConfig = cast(TRLConfig, DummyRLConfig()) if config is None else config
        self.setup()

    def setup(self) -> None:
        pass  # NotImplemented

    @abstractmethod
    def call_restore(self, data: Any, from_serialized: bool = False, from_worker: bool = False, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def call_backup(self, serialized: bool = False, to_worker: bool = False, **kwargs) -> Any:
        raise NotImplementedError()

    def restore(self, data: Any, from_serialized: bool = False, from_worker: bool = False, **kwargs) -> None:
        self.call_restore(data, from_serialized=from_serialized, from_worker=from_worker, **kwargs)

    def backup(self, serialized: bool = False, to_worker: bool = False, **kwargs) -> Any:
        dat = self.call_backup(serialized=serialized, to_worker=to_worker, **kwargs)
        return None if dat == [] else dat

    def save(self, path: str, compress: bool = True, **kwargs) -> None:
        logger.debug(f"parameter save: {path}")
        t0 = time.time()
        dat = self.backup(**kwargs)
        save_file(path, dat, compress)
        file_size = os.path.getsize(path)
        logger.info(f"parameter saved({file_size} bytes, {time.time() - t0:.1f}s): {os.path.basename(path)}")

    def load(self, path: str, **kwargs) -> None:
        logger.debug(f"parameter load: {path}")
        t0 = time.time()
        dat = load_file(path)
        self.restore(dat, **kwargs)
        logger.info(f"parameter loaded({time.time() - t0:.1f}s)")

    def summary(self, **kwargs):
        pass  # NotImplemented

    def update_from_worker_parameter(self, worker_parameger: "RLParameter") -> None:
        """WorkerからParameterの更新がある場合、最後の結果を保存する処理を記載
        - Configのuse_update_parameter_from_workerをTrueにする必要あり
        - 分散学習の最後でのみ呼ばれます
        """
        pass  # NotImplemented


class DummyRLParameter(RLParameter):
    def __init__(self, config: Optional[RLConfig] = None):
        super().__init__(config)

    def call_restore(self, data: Any, **kwargs) -> None:
        pass

    def call_backup(self, **kwargs) -> Any:
        return None
