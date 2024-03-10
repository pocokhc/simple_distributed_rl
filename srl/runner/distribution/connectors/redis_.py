import logging
import pickle
import traceback
import zlib
from typing import Any, List, Optional, Union, cast

import redis

from srl.runner.distribution.connectors.parameters import RedisParameters
from srl.runner.distribution.interface import IMemoryReceiver, IMemorySender, IParameterReader, IParameterWriter

logger = logging.getLogger(__name__)


class RedisConnector(IParameterWriter, IParameterReader, IMemoryReceiver, IMemorySender):
    def __init__(self, parameter: RedisParameters):
        self.parameter = parameter
        self.server = None
        self.queue_name = parameter.queue_name

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
                **self.parameter.kwargs,
            )
        else:
            self.server = redis.from_url(self.parameter.url, **self.parameter.kwargs)

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
        value = self.server.get(key)
        return cast(Optional[bytes], value)

    def set(self, key: str, value: Union[str, bytes]) -> bool:
        self.connect()
        assert self.server is not None
        return cast(bool, self.server.set(key, value))

    def exists(self, key: str) -> bool:
        self.connect()
        assert self.server is not None
        return cast(bool, self.server.exists(key))

    def get_keys(self, key: str) -> List[str]:
        self.connect()
        assert self.server is not None
        return [key for key in self.server.scan_iter(key)]

    def delete(self, key: str) -> None:
        self.connect()
        assert self.server is not None
        if self.exists(key):
            self.server.delete(key)

    def rpush(self, key: str, value: str) -> None:
        self.connect()
        assert self.server is not None
        self.server.rpush(key, value)

    # --- IParameterWriter
    def parameter_update(self, params: Any):
        try:
            if self.server is None:
                self.connect()
                assert self.server is not None
            self.server.set(self.parameter.parameter_name, zlib.compress(pickle.dumps(params)))
        except redis.RedisError:
            self.close()
            raise

    # --- IParameterReader
    def parameter_read(self) -> Optional[Any]:
        try:
            if self.server is None:
                self.connect()
                assert self.server is not None
            params = self.get(self.parameter.parameter_name)
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
            dat = self.server.lpop(self.queue_name)
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
            if self.exists(self.queue_name):
                self.server.delete(self.queue_name)
        except redis.RedisError:
            logger.error(traceback.format_exc())
            self.close()
        except Exception:
            logger.error(traceback.format_exc())

    # --- IMemorySender
    def memory_add(self, dat: Any) -> None:
        try:
            if self.server is None:
                self.connect()
                assert self.server is not None
            dat = pickle.dumps(dat)
            self.server.rpush(self.queue_name, dat)
        except redis.RedisError:
            self.close()
            raise

    def memory_size(self) -> int:
        try:
            if self.server is None:
                self.connect()
                if self.server is None:
                    return -1
            return cast(int, self.server.llen(self.queue_name))
        except redis.RedisError:
            logger.error(traceback.format_exc())
            self.close()
        except Exception:
            logger.error(traceback.format_exc())
        return -1
