import datetime
import logging
import pickle
import time
import uuid
from typing import Any, Optional, Tuple, cast

from srl.runner.distribution.connectors.imemory import IMemoryConnector, IServerParameters
from srl.runner.distribution.connectors.parameters import RedisParameters
from srl.runner.runner import TaskConfig

logger = logging.getLogger(__name__)


class DistributedManager:
    def __init__(self, parameter: RedisParameters, memory_parameter: Optional[IServerParameters]):
        from srl.runner.distribution.connectors.redis_ import RedisConnector

        self.parameter = parameter
        self.memory_parameter = memory_parameter

        self.server = cast(RedisConnector, parameter.create_memory_connector())
        self._keepalive_t0 = 0

    def set_user(self, role: str, uid: str = "", actor_idx: int = 0):
        self.role = role
        self.uid = str(uuid.uuid4()) if uid == "" else uid
        self.actor_idx = actor_idx

        logger.info(f"uid: {self.uid}")

    def create_args(self):
        return (
            self.parameter,
            self.memory_parameter,
            self.uid,
            self.role,
            self.actor_idx,
        )

    @staticmethod
    def create(
        parameter,
        memory_parameter,
        uid,
        role,
        actor_idx,
    ):
        m = DistributedManager(parameter, memory_parameter)
        m.uid = uid
        m.role = role
        m.actor_idx = actor_idx
        return m

    def create_memory_connector(self) -> IMemoryConnector:
        if self.memory_parameter is None:
            return self.server

        server = self.memory_parameter.create_memory_connector()
        server.ping()
        logger.info(f"create_memory_connector: {server}")
        return server

    def ping(self) -> bool:
        if not self.server.ping():
            return False
        if not self.create_memory_connector().ping():
            return False
        return True

    # -----------------------------
    # parameter
    # -----------------------------
    def parameter_read(self):
        params = self.server.server_get("task:parameter")
        return params if params is None else pickle.loads(params)

    def parameter_update(self, parameter: Any):
        self.server.server_set("task:parameter", pickle.dumps(parameter))

    # -----------------------------
    # task
    # -----------------------------
    def task_create(self, actor_num: int, task_config: Any, parameter: Any) -> None:
        self.server.server_set("task:status", "WAIT")
        self.server.server_set("task:config", pickle.dumps(task_config))
        self.server.server_set("task:actor_num", str(actor_num))
        for k in self.server.server_get_keys("task:trainer:*"):
            self.server.server_delete(k)
        self.server.server_set("task:trainer:id", "NO_ASSIGN")
        for i in range(actor_num):
            for k in self.server.server_get_keys(f"task:actor:{i}:*"):
                self.server.server_delete(k)
            self.server.server_set(f"task:actor:{i}:id", "NO_ASSIGN")
        self.server.server_set("task:parameter", pickle.dumps(parameter))
        logger.info("create new task")
        self.task_log("Create new task")

    def task_end(self):
        self.server.server_set("task:status", "END")
        logger.info("task end")
        self.task_log("End")

    def task_is_dead(self) -> bool:
        body = self.server.server_get("task:status")
        status = "" if body is None else body.decode()
        return status == "END"

    def task_get_status(self) -> str:
        body = self.server.server_get("task:status")
        return "" if body is None else body.decode()

    def task_get_actor_num(self) -> int:
        body = self.server.server_get("task:actor_num")
        if body is None:
            return 1
        return int(body.decode())

    def task_get_trainer(self, key: str) -> str:
        body = self.server.server_get(f"task:trainer:{key}")
        return "" if body is None else body.decode()

    def task_set_trainer(self, key: str, value: str) -> None:
        self.server.server_set(f"task:trainer:{key}", value)

    def task_get_actor(self, idx: int, key: str) -> str:
        body = self.server.server_get(f"task:actor:{idx}:{key}")
        return "" if body is None else body.decode()

    def task_set_actor(self, idx: int, key: str, value: str) -> None:
        self.server.server_set(f"task:actor:{idx}:{key}", value)

    def task_get_config(self) -> Optional[TaskConfig]:
        config = self.server.server_get("task:config")
        assert config is not None
        config = pickle.loads(config)
        return cast(TaskConfig, config)

    def task_assign_by_my_id(self) -> Tuple[bool, int]:
        now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.server.connect()
        assert self.server.server is not None

        status = self.task_get_status()
        if status != "WAIT":
            return False, 0

        # --- trainer assign check
        if self.role == "trainer":
            _tid = self.task_get_trainer("id")
            if _tid != "NO_ASSIGN":
                return False, 0

            self.server.server_set("task:trainer:id", self.uid)
            self.server.server_set("task:trainer:health", now_str)
            logger.info(f"Trainer assigned({self.uid})")
            self.task_log("Trainer assigned")
            return True, 0

        # --- actor assign check
        if self.role == "actor":
            actor_num = self.task_get_actor_num()
            for i in range(actor_num):
                _aid = self.task_get_actor(i, "id")
                if _aid != "NO_ASSIGN":
                    continue
                self.actor_idx = i
                self.server.server_set(f"task:actor:{i}:id", self.uid)
                self.server.server_set(f"task:actor:{i}:health", now_str)
                logger.info(f"Actor{i} assigned({self.uid})")
                self.task_log(f"Actor{i} assigned")
                return True, i

        return False, 0

    def _task_log_push(self, key: str, value: str):
        self.server.connect()
        assert self.server.server is not None
        self.server.server.rpush(key, value)

    def task_log(self, msg: str) -> None:
        now = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        msg = f"{now} [{self.role}] {msg} ({self.uid})"
        self._task_log_push("task:logs", msg)

    def keepalive(self, do_now: bool = False) -> bool:
        if not do_now and (time.time() - self._keepalive_t0 < self.parameter.keepalive_interval):
            return False
        self._keepalive_t0 = time.time()

        # --- update health
        now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if self.role == "trainer":
            # 一旦TODO
            # _tid = self.task_get_trainer("id")
            # assert _tid == self.uid
            self.task_set_trainer("health", now_str)
        elif self.role == "actor":
            # _aid = self.task_get_actor(self.actor_idx, "id")
            # assert _aid == self.uid
            self.task_set_actor(self.actor_idx, "health", now_str)
        elif self.role == "client":
            # --- 全部アサインされたらRUNに変更
            if self.task_get_status() == "WAIT":
                is_all_assigned = True
                _tid = self.task_get_trainer("id")
                if _tid == "" or _tid == "NO_ASSIGN":
                    is_all_assigned = False
                for idx in range(self.task_get_actor_num()):
                    _aid = self.task_get_actor(idx, "id")
                    if _aid == "" or _aid == "NO_ASSIGN":
                        is_all_assigned = False
                if is_all_assigned:
                    self.server.server_set("task:status", "RUNNING")

        return True
