import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar, cast

from srl.base.rl.config import DummyRLConfig, RLConfig, TRLConfig

logger = logging.getLogger(__name__)

TRLParameter = TypeVar("TRLParameter", bound="RLParameter", covariant=True)


class RLParameter(ABC, Generic[TRLConfig]):
    def __init__(self, config: Optional[TRLConfig]):
        self.config: TRLConfig = cast(TRLConfig, DummyRLConfig()) if config is None else config
        self.setup()

    def setup(self) -> None:
        pass  # NotImplemented

    @abstractmethod
    def call_restore(self, data: Any, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def call_backup(self, **kwargs) -> Any:
        raise NotImplementedError()

    def restore(self, data: Any, **kwargs) -> None:
        self.call_restore(data, **kwargs)

    def backup(self, **kwargs) -> Any:
        dat = self.call_backup(**kwargs)
        return None if dat == [] else dat

    def save(self, path: str) -> None:
        logger.debug(f"parameter save: {path}")
        try:
            t0 = time.time()
            with open(path, "wb") as f:
                pickle.dump(self.backup(), f)

            file_size = os.path.getsize(path)
            logger.info(f"parameter saved({file_size} bytes, {time.time() - t0:.1f}s): {os.path.basename(path)}")
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
        super().__init__(config)

    def call_restore(self, data: Any, **kwargs) -> None:
        pass

    def call_backup(self, **kwargs) -> Any:
        return None
