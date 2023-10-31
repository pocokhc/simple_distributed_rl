import datetime
import logging
import pickle
import time
import traceback
import uuid
from typing import Any, Optional, Tuple, Union, cast

import redis

logger = logging.getLogger(__name__)


class DistributedManager:
    def __init__(
        self,
        host: str,
        redis_kwargs: dict = {},
        keepalive_interval: int = 10,
    ):
        self.host = host
        self.redis_kwargs = redis_kwargs
        self.keepalive_interval = keepalive_interval

        self.server = None
        self._keepalive_t0 = 0

    def set_user(self, role: str, uid: str = "", actor_idx: int = 0):
        self.role = role
        self.uid = str(uuid.uuid4()) if uid == "" else uid
        self.actor_idx = actor_idx

        logger.info(f"uid: {self.uid}")

    def __del__(self):
        self.close()

    def create_args(self):
        return (
            self.host,
            self.redis_kwargs,
            self.keepalive_interval,
            self.uid,
            self.role,
            self.actor_idx,
        )

    @staticmethod
    def create(
        host,
        redis_kwargs,
        keepalive_interval,
        uid,
        role,
        actor_idx,
    ):
        m = DistributedManager(host, redis_kwargs, keepalive_interval)
        m.uid = uid
        m.role = role
        m.actor_idx = actor_idx
        return m

    def close(self):
        if self.server is not None:
            try:
                self.server.close()
            except Exception:
                logger.error(traceback.format_exc())

        self.server = None

    def server_ping(self):
        if self.server is None:
            self.server = redis.Redis(self.host, **self.redis_kwargs)
        self.server.ping()

    def connect(self) -> bool:
        if self.server is not None:
            return True
        try:
            self.server = redis.Redis(self.host, **self.redis_kwargs)
            return True
        except Exception:
            logger.error(traceback.format_exc())
            self.server = None
        return False

    def server_get(self, key: str) -> Optional[bytes]:
        if not self.connect():
            return None
        assert self.server is not None
        value = self.server.get(key)
        return cast(Optional[bytes], value)

    def server_set(self, key: str, value: Union[str, bytes]) -> bool:
        if not self.connect():
            return False
        assert self.server is not None
        return cast(bool, self.server.set(key, value))

    def server_exists(self, key: str) -> bool:
        if not self.connect():
            return False
        assert self.server is not None
        return cast(bool, self.server.exists(key))

    # -----------------------------
    # memory
    # -----------------------------
    def memory_add(self, task_id: str, dat: Any):
        try:
            if not self.connect():
                return False
            assert self.server is not None
            dat = pickle.dumps(dat)
            self.server.rpush(f"task:{task_id}:memory", dat)
        except Exception:
            logger.error(traceback.format_exc())
            self.close()
        return None

    def memory_recv(self, task_id: str) -> Optional[Any]:
        try:
            if not self.connect():
                return None
            assert self.server is not None
            dat = self.server.lpop(f"task:{task_id}:memory")
            return dat if dat is None else pickle.loads(cast(bytes, dat))
        except Exception:
            logger.error(traceback.format_exc())
            self.close()
        return None

    def memory_size(self, task_id: str) -> int:
        try:
            if not self.connect():
                return -1
            assert self.server is not None
            return cast(int, self.server.llen(f"task:{task_id}:memory"))
        except Exception:
            logger.error(traceback.format_exc())
            self.close()
        return -1

    def memory_reset(self, task_id: str) -> bool:
        try:
            if not self.connect():
                return False
            assert self.server is not None
            key = f"task:{task_id}:memory"
            for _ in range(cast(int, self.server.llen(key))):
                self.server.lpop(key)
            return True
        except Exception:
            logger.error(traceback.format_exc())
            self.close()
        return False

    # -----------------------------
    # parameter
    # -----------------------------
    def parameter_read(self, task_id: str = ""):
        if task_id == "":
            task_id = self.uid
        params = self.server_get(f"task:{task_id}:parameter")
        return params if params is None else pickle.loads(params)

    def parameter_update(self, task_id: str, parameter: Any):
        # 自分のidが登録されている場合のみ更新
        _tid = self.task_get_trainer(task_id, "id")
        assert _tid == self.uid
        self.server_set(f"task:{task_id}:parameter", pickle.dumps(parameter))

    # -----------------------------
    # task
    # -----------------------------
    def task_add(self, actor_num: int, task_config: Any, parameter: Any) -> str:
        task_id = self.uid  # とりあえず 1client=1task
        self.server_set(f"task:{task_id}:status", "WAIT")
        self.server_set(f"task:{task_id}:config", pickle.dumps(task_config))
        self.server_set(f"task:{task_id}:actor_num", str(actor_num))
        for i in range(actor_num):
            self.server_set(f"task:{task_id}:actor:{i}:id", "NO_ASSIGN")
        self.server_set(f"task:{task_id}:trainer:id", "NO_ASSIGN")
        self.server_set(f"task:{task_id}:parameter", pickle.dumps(parameter))
        self.server_set(f"taskid:{task_id}", task_id)  # last
        logger.info(f"add task: {task_id}")

        self.log(task_id, "create task")
        return task_id

    def task_end(self, task_id: str):
        if not self.server_exists(f"taskid:{task_id}"):
            return
        self.server_set(f"task:{task_id}:status", "END")
        logger.info(f"task end: {task_id}")
        self.log(task_id, "End")

    def task_is_dead(self, task_id: str) -> bool:
        if not self.server_exists(f"taskid:{task_id}"):
            return True
        body = self.server_get(f"task:{task_id}:status")
        status = "" if body is None else body.decode()
        return status == "END"

    def task_get_status(self, task_id: str) -> str:
        if not self.server_exists(f"taskid:{task_id}"):
            return ""
        body = self.server_get(f"task:{task_id}:status")
        return "" if body is None else body.decode()

    def task_get_actor_num(self, task_id: str) -> int:
        if not self.server_exists(f"taskid:{task_id}"):
            return -1
        body = self.server_get(f"task:{task_id}:actor_num")
        if body is None:
            return 1
        return int(body.decode())

    def task_get_trainer(self, task_id: str, key: str) -> str:
        if not self.server_exists(f"taskid:{task_id}"):
            return ""
        body = self.server_get(f"task:{task_id}:trainer:{key}")
        return "" if body is None else body.decode()

    def task_set_trainer(self, task_id: str, key: str, value: str) -> None:
        if not self.server_exists(f"taskid:{task_id}"):
            return
        self.server_set(f"task:{task_id}:trainer:{key}", value)

    def task_get_actor(self, task_id: str, idx: int, key: str) -> str:
        if not self.server_exists(f"taskid:{task_id}"):
            return ""
        body = self.server_get(f"task:{task_id}:actor:{idx}:{key}")
        return "" if body is None else body.decode()

    def task_set_actor(self, task_id: str, idx: int, key: str, value: str) -> None:
        if not self.server_exists(f"taskid:{task_id}"):
            return
        self.server_set(f"task:{task_id}:actor:{idx}:{key}", value)

    def task_assign_by_my_id(self) -> Tuple[str, Optional[Any], int]:
        now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.connect()
        assert self.server is not None
        keys = cast(list, self.server.keys("taskid:*"))
        for key in keys:
            task_id = self.server_get(key)
            assert task_id is not None
            task_id = task_id.decode()
            status = self.task_get_status(task_id)
            if status != "WAIT":
                continue

            # --- trainer assign check
            if self.role == "trainer":
                _tid = self.task_get_trainer(task_id, "id")
                if _tid != "NO_ASSIGN":
                    continue
                config = self.server_get(f"task:{task_id}:config")
                assert config is not None
                config = pickle.loads(config)

                self.server_set(f"task:{task_id}:trainer:id", self.uid)
                self.server_set(f"task:{task_id}:trainer:health", now_str)
                logger.info(f"Trainer assigned({self.uid}): {task_id}")
                self.log(task_id, "Trainer assigned")
                return task_id, config, 0

            # --- actor assign check
            if self.role == "actor":
                actor_num = self.task_get_actor_num(task_id)
                for i in range(actor_num):
                    _aid = self.task_get_actor(task_id, i, "id")
                    if _aid != "NO_ASSIGN":
                        continue
                    config = self.server_get(f"task:{task_id}:config")
                    assert config is not None
                    config = pickle.loads(config)

                    self.actor_idx = i
                    self.server_set(f"task:{task_id}:actor:{i}:id", self.uid)
                    self.server_set(f"task:{task_id}:actor:{i}:health", now_str)
                    logger.info(f"Actor{i} assigned({self.uid}): {task_id}")
                    self.log(task_id, f"Actor{i} assigned")
                    return task_id, config, i

        return "", None, 0

    def _log_push(self, key: str, value: str):
        if not self.connect():
            return False
        assert self.server is not None
        self.server.rpush(key, value)

    def log(self, task_id: str, msg: str) -> None:
        if not self.server_exists(f"taskid:{task_id}"):
            return
        now = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        msg = f"{now} [{self.role}] {msg} ({self.uid})"
        self._log_push(f"task:{task_id}:logs", msg)

    def keepalive(self, task_id: str) -> bool:
        if not self.server_exists(f"taskid:{task_id}"):
            return False

        if time.time() - self._keepalive_t0 > self.keepalive_interval:
            self._keepalive_t0 = time.time()

            # --- update health
            now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            if self.role == "trainer":
                _tid = self.task_get_trainer(task_id, "id")
                assert _tid == self.uid
                self.task_set_trainer(task_id, "health", now_str)
            elif self.role == "actor":
                _aid = self.task_get_actor(task_id, self.actor_idx, "id")
                assert _aid == self.uid
                self.task_set_actor(task_id, self.actor_idx, "health", now_str)

            return True
        return False
