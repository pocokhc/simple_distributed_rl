import logging
import pickle
import traceback
from typing import Any, Optional, Union, cast

import redis

from srl.runner.distribution.connectors.imemory import IMemoryConnector
from srl.runner.distribution.connectors.parameters import RedisParameters

logger = logging.getLogger(__name__)


class RedisConnector(IMemoryConnector):
    def __init__(self, parameter: RedisParameters):
        self.parameter = parameter
        self.server = None

    def __del__(self):
        self.close()

    def close(self):
        if self.server is not None:
            try:
                self.server.close()
            except Exception:
                logger.error(traceback.format_exc())

        self.server = None

    def connect(self) -> None:
        if self.server is not None:
            return
        if self.parameter.url == "":
            self.server = redis.Redis(self.parameter.host, **self.parameter.kwargs)
        else:
            self.server = redis.from_url(self.parameter.url, **self.parameter.kwargs)

    def ping(self) -> bool:
        self.connect()
        assert self.server is not None
        return cast(bool, self.server.ping())

    def server_get(self, key: str) -> Optional[bytes]:
        self.connect()
        assert self.server is not None
        value = self.server.get(key)
        return cast(Optional[bytes], value)

    def server_set(self, key: str, value: Union[str, bytes]) -> bool:
        self.connect()
        assert self.server is not None
        return cast(bool, self.server.set(key, value))

    def server_exists(self, key: str) -> bool:
        self.connect()
        assert self.server is not None
        return cast(bool, self.server.exists(key))

    # -----------------------------
    # memory
    # -----------------------------
    def memory_setup(self, task_id: str) -> None:
        self.memory_name = f"task:{task_id}:memory"
        self.connect()

    def memory_add(self, dat: Any) -> bool:
        try:
            assert self.server is not None
            dat = pickle.dumps(dat)
            self.server.rpush(self.memory_name, dat)
            return True
        except Exception:
            logger.error(traceback.format_exc())
            self.close()
            self.connect()
        return False

    def memory_recv(self) -> Optional[Any]:
        try:
            assert self.server is not None
            dat = self.server.lpop(self.memory_name)
            return dat if dat is None else pickle.loads(cast(bytes, dat))
        except Exception:
            logger.error(traceback.format_exc())
            self.close()
            self.connect()
        return None

    def memory_size(self) -> int:
        try:
            assert self.server is not None
            return cast(int, self.server.llen(self.memory_name))
        except Exception:
            logger.error(traceback.format_exc())
            self.close()
            self.connect()
        return 0

    def memory_delete_if_exist(self, task_id: str) -> None:
        if not self.connect():
            return
        assert self.server is not None
        m_name = f"task:{task_id}:memory"
        if self.server.exists(m_name):
            self.server.delete(m_name)

    def memory_exist(self, task_id: str) -> bool:
        if not self.connect():
            return False
        assert self.server is not None
        return cast(bool, self.server.exists(f"task:{task_id}:memory"))
