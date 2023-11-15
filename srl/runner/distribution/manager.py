import datetime
import logging
import pickle
import random
import time
import uuid
from typing import Any, Optional, Tuple, cast

import srl
from srl.base.exception import DistributionError
from srl.runner.distribution.connectors.imemory import IMemoryConnector, IServerParameters
from srl.runner.distribution.connectors.parameters import RedisParameters
from srl.runner.runner import TaskConfig
from srl.utils.common import compare_equal_version

logger = logging.getLogger(__name__)


class DistributedManager:
    def __init__(self, parameter: RedisParameters, memory_parameter: Optional[IServerParameters]):
        from srl.runner.distribution.connectors.redis_ import RedisConnector

        self.parameter = parameter
        self.memory_parameter = memory_parameter

        self.server = cast(RedisConnector, parameter.create_memory_connector())
        self._keepalive_t0 = 0
        self._memory_connector = None
        self.health_start_time = 0
        self.health_t0 = 0

        self.keepalive_threshold = self.parameter.keepalive_interval * 2.2
        if self.keepalive_threshold < 5:
            self.keepalive_threshold = 5

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
            self.health_start_time,
            self.health_t0,
        )

    @staticmethod
    def create(
        parameter,
        memory_parameter,
        uid,
        role,
        actor_idx,
        health_start_time,
        health_t0,
    ):
        m = DistributedManager(parameter, memory_parameter)
        m.uid = uid
        m.role = role
        m.actor_idx = actor_idx
        m.health_start_time = health_start_time
        m.health_t0 = health_t0
        return m

    def create_memory_connector(self) -> IMemoryConnector:
        if self.memory_parameter is None:
            return self.server
        if self._memory_connector is not None:
            return self._memory_connector

        self._memory_connector = self.memory_parameter.create_memory_connector()
        logger.info(f"create_memory_connector: {self._memory_connector}")
        return self._memory_connector

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
        """例外を出す"""
        params = self.server.server_get(f"{self.parameter.task_name}:parameter")
        return params if params is None else pickle.loads(params)

    def parameter_update(self, parameter: Any):
        """例外を出す"""
        self.server.server_set(f"{self.parameter.task_name}:parameter", pickle.dumps(parameter))

    # -----------------------------
    # task
    # -----------------------------
    def task_create(self, actor_num: int, task_config: Any, parameter: Any) -> None:
        self.health_t0 = time.time()
        self.server.server_set(f"{self.parameter.task_name}:version", srl.__version__)
        self.server.server_set(f"{self.parameter.task_name}:config", pickle.dumps(task_config))
        self.server.server_set(f"{self.parameter.task_name}:actor_num", str(actor_num))
        self.server.server_set(f"{self.parameter.task_name}:setup_queue", "0")
        self.server.server_set(f"{self.parameter.task_name}:health", "0")
        for k in self.server.server_get_keys(f"{self.parameter.task_name}:trainer:*"):
            self.server.server_delete(k)
        self.server.server_set(f"{self.parameter.task_name}:trainer:id", "")
        for i in range(actor_num):
            for k in self.server.server_get_keys(f"{self.parameter.task_name}:actor:{i}:*"):
                self.server.server_delete(k)
            self.server.server_set(f"{self.parameter.task_name}:actor:{i}:id", "")
        self.server.server_set(f"{self.parameter.task_name}:parameter", pickle.dumps(parameter))
        self.server.server_set(f"{self.parameter.task_name}:status", "WAIT")
        logger.info("create new task")
        self.task_log("Create new task")

    def task_end(self):
        self.server.server_set(f"{self.parameter.task_name}:status", "END")
        self.server.server_set(f"{self.parameter.task_name}:setup_queue", "0")
        logger.info("task end")
        self.task_log("End")

    def task_is_dead(self) -> bool:
        body = self.server.server_get(f"{self.parameter.task_name}:status")
        status = "" if body is None else body.decode()
        if status == "END":
            s = "Task is dead(status END)"
            self.task_log(s)
            logger.info(s)
            return True

        # health check
        task_time = self.task_get_task_time()
        if (time.time() - self.health_t0 + self.health_start_time) - task_time > self.keepalive_threshold:
            s = "Task is dead(health over)"
            self.task_log(s)
            logger.info(s)
            return True
        return False

    def task_get_status(self) -> str:
        body = self.server.server_get(f"{self.parameter.task_name}:status")
        return "" if body is None else body.decode()

    def task_get_actor_num(self) -> int:
        body = self.server.server_get(f"{self.parameter.task_name}:actor_num")
        if body is None:
            return 1
        return int(body.decode())

    def task_set_setup_queue(self) -> None:
        self.server.server_set(f"{self.parameter.task_name}:setup_queue", "1")

    def task_is_setup_queue(self) -> bool:
        body = self.server.server_get(f"{self.parameter.task_name}:setup_queue")
        if body is None:
            return False
        return body.decode() == "1"

    def task_get_task_time(self) -> float:
        body = self.server.server_get(f"{self.parameter.task_name}:health")
        if body is None:
            return 0
        return float(body.decode())

    def task_get_trainer(self, key: str) -> str:
        body = self.server.server_get(f"{self.parameter.task_name}:trainer:{key}")
        return "" if body is None else body.decode()

    def task_set_trainer(self, key: str, value: str) -> None:
        self.server.server_set(f"{self.parameter.task_name}:trainer:{key}", value)

    def task_get_actor(self, idx: int, key: str) -> str:
        body = self.server.server_get(f"{self.parameter.task_name}:actor:{idx}:{key}")
        return "" if body is None else body.decode()

    def task_set_actor(self, idx: int, key: str, value: str) -> None:
        self.server.server_set(f"{self.parameter.task_name}:actor:{idx}:{key}", value)

    def task_get_config(self) -> Optional[TaskConfig]:
        config = self.server.server_get(f"{self.parameter.task_name}:config")
        assert config is not None
        config = pickle.loads(config)
        return cast(TaskConfig, config)

    def task_assign_by_my_id(self) -> Tuple[bool, int]:
        status = self.task_get_status()
        if status != "WAIT":
            return False, 0

        # --- trainer assign check
        if self.role == "trainer":
            _tid = self.task_get_trainer("id")
            if _tid != "":
                return False, 0

            self.server.server_set(f"{self.parameter.task_name}:trainer:id", self.uid)
            self.server.server_set(f"{self.parameter.task_name}:trainer:health", "0")
            logger.info(f"Trainer assigned({self.uid})")
            self.task_log("Trainer assigned")
            self.health_start_time = self.task_get_task_time()
            self.health_t0 = time.time()
            self._check_version()
            return True, 0

        # --- actor assign check
        if self.role == "actor":
            actor_num = self.task_get_actor_num()
            for i in range(actor_num):
                _aid = self.task_get_actor(i, "id")
                if _aid != "":
                    continue
                self.actor_idx = i
                self.server.server_set(f"{self.parameter.task_name}:actor:{i}:id", self.uid)
                self.server.server_set(f"{self.parameter.task_name}:actor:{i}:health", "0")
                logger.info(f"Actor{i} assigned({self.uid})")
                self.task_log(f"Actor{i} assigned")
                self.health_start_time = self.task_get_task_time()
                self.health_t0 = time.time()
                self._check_version()
                return True, i

        return False, 0

    def _check_version(self):
        try:
            v = self.server.server_get(f"{self.parameter.task_name}:version")
            if v is None:
                return
            if not compare_equal_version(v.decode(), srl.__version__):
                logger.warning(f"SRL version is different({v} != {srl.__version__})")
        except Exception as e:
            logger.info(f"SRL version check fail: {e}")

    def _task_log_push(self, key: str, value: str):
        self.server.connect()
        assert self.server.server is not None
        self.server.server.rpush(key, value)

    def task_log(self, msg: str) -> None:
        now = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        msg = f"{now} [{self.role}] {msg} ({self.uid})"
        self._task_log_push(f"{self.parameter.task_name}:logs", msg)

    def keepalive_trainer(self, do_now: bool = False) -> bool:
        if not do_now and (time.time() - self._keepalive_t0 < self.parameter.keepalive_interval):
            return False
        self._keepalive_t0 = time.time()

        self.task_set_trainer("health", str(time.time() - self.health_t0 + self.health_start_time))
        return True

    def keepalive_actor(self, do_now: bool = False) -> bool:
        if not do_now and (time.time() - self._keepalive_t0 < self.parameter.keepalive_interval):
            return False
        self._keepalive_t0 = time.time()

        # --- 2重にアサインされていないかチェック
        aid = self.task_get_actor(self.actor_idx, "id")
        if aid != self.uid:
            # アサインされていたらランダム秒まって止める
            s = f"Another actor has been assigned. my:{self.uid}, another: {aid}"
            self.task_log(s)
            time.sleep(random.randint(0, 5))
            raise DistributionError(s)

        self.task_set_actor(self.actor_idx, "health", str(time.time() - self.health_t0 + self.health_start_time))
        return True

    def keepalive_client(self, do_now: bool = False) -> bool:
        if not do_now and (time.time() - self._keepalive_t0 < self.parameter.keepalive_interval):
            return False
        self._keepalive_t0 = time.time()

        task_time = time.time() - self.health_t0
        self.server.server_set(f"{self.parameter.task_name}:health", str(task_time))

        # --- health check trainer
        health = self.task_get_trainer("health")
        if health != "":
            diff_time = task_time - float(health)
            if diff_time > self.keepalive_threshold:
                tid = self.task_get_trainer("id")
                s = f"Trainer remove(health over) {diff_time:.1f}s {tid}"
                self.task_log(s)
                logger.info(s)
                self.task_set_trainer("id", "")
                self.task_set_trainer("health", "")

        # --- health check actor
        actor_num = self.task_get_actor_num()
        for i in range(actor_num):
            health = self.task_get_actor(i, "health")
            if health != "":
                diff_time = task_time - float(health)
                if diff_time > self.keepalive_threshold:
                    aid = self.task_get_actor(i, "id")
                    s = f"Actor{i} remove(health over) {diff_time:.1f}s {aid}"
                    self.task_log(s)
                    logger.info(s)
                    self.task_set_actor(i, "id", "")
                    self.task_set_actor(i, "health", "")

        # --- 全部アサインされたらRUNに変更
        status = self.task_get_status()
        if status in ["WAIT", "RUN"]:
            is_all_assigned = True
            _tid = self.task_get_trainer("id")
            if _tid == "" or _tid == "":
                is_all_assigned = False
            for idx in range(self.task_get_actor_num()):
                _aid = self.task_get_actor(idx, "id")
                if _aid == "" or _aid == "":
                    is_all_assigned = False
            if status == "WAIT":
                if is_all_assigned:
                    s = "change status: RUN"
                    self.task_log(s)
                    logger.info(s)
                    self.server.server_set(f"{self.parameter.task_name}:status", "RUN")
            elif status == "RUN":
                if not is_all_assigned:
                    s = "change status: WAIT"
                    self.task_log(s)
                    logger.info(s)
                    self.server.server_set(f"{self.parameter.task_name}:status", "WAIT")

        return True
