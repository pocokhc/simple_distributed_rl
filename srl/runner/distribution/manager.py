import datetime
import logging
import pickle
import time
import uuid
from typing import Any, List, Optional, Tuple, cast

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
    def parameter_read(self, task_id: str = ""):
        if task_id == "":
            task_id = self.uid
        params = self.server.server_get(f"task:{task_id}:parameter")
        return params if params is None else pickle.loads(params)

    def parameter_update(self, task_id: str, parameter: Any) -> bool:
        # 自分のidが登録されている場合のみ更新
        _tid = self.task_get_trainer(task_id, "id")
        assert _tid == self.uid
        self.server.server_set(f"task:{task_id}:parameter", pickle.dumps(parameter))
        return True

    # -----------------------------
    # task
    # -----------------------------
    def task_add(self, actor_num: int, task_config: Any, parameter: Any) -> str:
        task_id = self.uid  # とりあえず 1client=1task
        self.server.server_set(f"task:{task_id}:status", "WAIT")
        self.server.server_set(f"task:{task_id}:config", pickle.dumps(task_config))
        self.server.server_set(f"task:{task_id}:actor_num", str(actor_num))
        for i in range(actor_num):
            self.server.server_set(f"task:{task_id}:actor:{i}:id", "NO_ASSIGN")
        self.server.server_set(f"task:{task_id}:trainer:id", "NO_ASSIGN")
        self.server.server_set(f"task:{task_id}:parameter", pickle.dumps(parameter))
        self.server.server_set(f"taskid:{task_id}", task_id)  # last
        logger.info(f"add task: {task_id}")

        self.task_log(task_id, "create task")
        return task_id

    def task_end(self, task_id: str):
        if not self.server.server_exists(f"taskid:{task_id}"):
            return
        self.server.server_set(f"task:{task_id}:status", "END")
        logger.info(f"task end: {task_id}")
        self.task_log(task_id, "End")

    def task_is_dead(self, task_id: str) -> bool:
        if not self.server.server_exists(f"taskid:{task_id}"):
            return True
        body = self.server.server_get(f"task:{task_id}:status")
        status = "" if body is None else body.decode()
        return status == "END"

    def task_get_status(self, task_id: str) -> str:
        if not self.server.server_exists(f"taskid:{task_id}"):
            return ""
        body = self.server.server_get(f"task:{task_id}:status")
        return "" if body is None else body.decode()

    def task_get_actor_num(self, task_id: str) -> int:
        if not self.server.server_exists(f"taskid:{task_id}"):
            return -1
        body = self.server.server_get(f"task:{task_id}:actor_num")
        if body is None:
            return 1
        return int(body.decode())

    def task_get_trainer(self, task_id: str, key: str) -> str:
        if not self.server.server_exists(f"taskid:{task_id}"):
            return ""
        body = self.server.server_get(f"task:{task_id}:trainer:{key}")
        return "" if body is None else body.decode()

    def task_set_trainer(self, task_id: str, key: str, value: str) -> None:
        if not self.server.server_exists(f"taskid:{task_id}"):
            return
        self.server.server_set(f"task:{task_id}:trainer:{key}", value)

    def task_get_actor(self, task_id: str, idx: int, key: str) -> str:
        if not self.server.server_exists(f"taskid:{task_id}"):
            return ""
        body = self.server.server_get(f"task:{task_id}:actor:{idx}:{key}")
        return "" if body is None else body.decode()

    def task_set_actor(self, task_id: str, idx: int, key: str, value: str) -> None:
        if not self.server.server_exists(f"taskid:{task_id}"):
            return
        self.server.server_set(f"task:{task_id}:actor:{idx}:{key}", value)

    def task_get_config(self, task_id: str) -> Optional[TaskConfig]:
        config = self.server.server_get(f"task:{task_id}:config")
        assert config is not None
        config = pickle.loads(config)
        return cast(TaskConfig, config)

    def task_get_ids(self) -> List[str]:
        self.server.connect()
        assert self.server.server is not None

        task_ids = []
        for key in self.server.server.scan_iter("taskid:*"):
            task_id = self.server.server_get(key)
            if task_id is not None:
                task_ids.append(task_id.decode())
        return task_ids

    def task_assign_by_my_id(self) -> Tuple[str, int]:
        now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.server.connect()
        assert self.server.server is not None

        keys = cast(list, self.server.server.keys("taskid:*"))
        for key in keys:
            task_id = self.server.server_get(key)
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

                self.server.server_set(f"task:{task_id}:trainer:id", self.uid)
                self.server.server_set(f"task:{task_id}:trainer:health", now_str)
                logger.info(f"Trainer assigned({self.uid}): {task_id}")
                self.task_log(task_id, "Trainer assigned")
                return task_id, 0

            # --- actor assign check
            if self.role == "actor":
                actor_num = self.task_get_actor_num(task_id)
                for i in range(actor_num):
                    _aid = self.task_get_actor(task_id, i, "id")
                    if _aid != "NO_ASSIGN":
                        continue
                    self.actor_idx = i
                    self.server.server_set(f"task:{task_id}:actor:{i}:id", self.uid)
                    self.server.server_set(f"task:{task_id}:actor:{i}:health", now_str)
                    logger.info(f"Actor{i} assigned({self.uid}): {task_id}")
                    self.task_log(task_id, f"Actor{i} assigned")
                    return task_id, i

        return "", 0

    def _task_log_push(self, key: str, value: str):
        self.server.connect()
        assert self.server.server is not None
        self.server.server.rpush(key, value)

    def task_log(self, task_id: str, msg: str) -> None:
        if not self.server.server_exists(f"taskid:{task_id}"):
            return
        now = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        msg = f"{now} [{self.role}] {msg} ({self.uid})"
        self._task_log_push(f"task:{task_id}:logs", msg)

    def keepalive(self, task_id: str, do_now: bool = False) -> bool:
        if not self.server.server_exists(f"taskid:{task_id}"):
            return False

        if not do_now and time.time() - self._keepalive_t0 < self.parameter.keepalive_interval:
            return False
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

    def server_cleaning(self):
        self.server.connect()
        assert self.server.server is not None

        memory_manager = self.create_memory_connector()
        for id in self.task_get_ids():
            status = self.task_get_status(id)
            if status != "END":
                continue

            # --- delete queue
            memory_manager.memory_delete_if_exist(id)

            # --- delete key
            keys = cast(list, self.server.server.keys(match=f"task:{id}:*"))
            self.server.server.delete(*keys)

            # すべて消えてたらidも消す
            keys = cast(list, self.server.server.keys(match=f"task:{id}:*"))
            if not memory_manager.memory_exist(id) and len(keys) == 0:
                self.server.server.delete(f"taskid:{id}")
                logger.info(f"task deleted: {id}")
