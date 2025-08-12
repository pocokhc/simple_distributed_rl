import logging
import pickle
import traceback
import zlib
from typing import Any, List, Optional, Union, cast

import redis

from srl.runner.distribution.connector_configs import IMemoryReceiver, IMemorySender, IParameterServer, RedisParameters

logger = logging.getLogger(__name__)


class RedisConnector(IParameterServer, IMemoryReceiver, IMemorySender):
    def __init__(self, parameter: RedisParameters):
        self.parameter = parameter
        self.server = None
        self._task_key = parameter.task_key + ":"
        self._parameter_key = parameter.parameter_key
        self._queue_key = parameter.queue_key

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
            self.server = redis.Redis(
                self.parameter.host,
                self.parameter.port,
                self.parameter.db,
                **self.parameter.redis_kwargs,
            )
        else:
            self.server = redis.from_url(self.parameter.url, **self.parameter.redis_kwargs)

    @property
    def is_connected(self) -> bool:
        return self.server is not None

    def ping(self) -> bool:
        try:
            self.connect()
            if self.server is not None:
                return cast(bool, self.server.ping())
        except redis.RedisError as e:
            logger.error(e)
            self.close()
        except Exception as e:
            logger.error(e)
        return False

    def get(self, key: str) -> Optional[bytes]:
        self.connect()
        assert self.server is not None
        value = self.server.get(self._task_key + key)
        return cast(Optional[bytes], value)

    def set(self, key: str, value: Union[str, bytes]) -> bool:
        self.connect()
        assert self.server is not None
        return cast(bool, self.server.set(self._task_key + key, value))

    def exists(self, key: str) -> bool:
        self.connect()
        assert self.server is not None
        return cast(bool, self.server.exists(self._task_key + key))

    def get_keys(self, key: str) -> List[str]:
        self.connect()
        assert self.server is not None
        keys = [
            k.decode("utf-8") if isinstance(k, bytes) else k  #
            for k in self.server.scan_iter(self._task_key + key)
        ]
        return [k[len(self._task_key) :] for k in keys]

    def delete(self, key: str) -> None:
        self.connect()
        assert self.server is not None
        if self.exists(key):
            self.server.delete(self._task_key + key)

    def rpush(self, key: str, value: str) -> None:
        self.connect()
        assert self.server is not None
        self.server.rpush(self._task_key + key, value)

    # --- IParameterServer
    def parameter_write(self, param_dat: Optional[Any], init: bool = False):
        try:
            if param_dat is None:
                if not init:
                    return
            if self.server is None:
                self.connect()
                assert self.server is not None
            self.server.set(self._task_key + self._parameter_key, zlib.compress(pickle.dumps(param_dat)))
        except redis.RedisError:
            self.close()
            raise

    def parameter_read(self) -> Optional[Any]:
        try:
            if self.server is None:
                self.connect()
                assert self.server is not None
            params = self.get(self._parameter_key)
            return params if params is None else pickle.loads(zlib.decompress(params))
        except redis.RedisError:
            self.close()
            raise

    # --- IMemoryReceiver
    def memory_recv(self) -> Optional[Any]:
        try:
            if self.server is None:
                self.connect()
                assert self.server is not None
            dat = self.server.lpop(self._task_key + self._queue_key)
            return dat if dat is None else pickle.loads(cast(bytes, dat))
        except redis.RedisError:
            self.close()
            raise

    def memory_purge(self) -> None:
        try:
            if self.server is None:
                self.connect()
                if self.server is None:
                    return
            if self.exists(self._queue_key):
                self.server.delete(self._task_key + self._queue_key)
        except redis.RedisError:
            logger.error(traceback.format_exc())
            self.close()
        except Exception:
            logger.error(traceback.format_exc())

    # --- IMemorySender
    def memory_send(self, dat: Any) -> None:
        try:
            if self.server is None:
                self.connect()
                assert self.server is not None
            dat = pickle.dumps(dat)
            self.server.rpush(self._task_key + self._queue_key, dat)
        except redis.RedisError:
            self.close()
            raise

    def memory_size(self) -> int:
        try:
            if self.server is None:
                self.connect()
                if self.server is None:
                    return -1
            return cast(int, self.server.llen(self._task_key + self._queue_key))
        except redis.RedisError:
            logger.error(traceback.format_exc())
            self.close()
        except Exception:
            logger.error(traceback.format_exc())
        return -1
