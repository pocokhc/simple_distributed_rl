import datetime
import logging
import pickle
import uuid
from dataclasses import dataclass
from typing import Any, Optional, cast

import srl
from srl.runner.distribution.connectors.parameters import RedisParameters
from srl.runner.distribution.connectors.redis_ import RedisConnector
from srl.runner.distribution.interface import IMemoryReceiver
from srl.runner.runner import TaskConfig
from srl.utils.common import compare_equal_version

logger = logging.getLogger(__name__)


@dataclass
class TaskManagerParams:
    role: str
    keepalive_interval: int = 10
    keepalive_threshold: int = 101
    uid: str = ""
    used_device_tf: str = "/CPU"
    used_device_torch: str = "cpu"

    def __post_init__(self):
        self.task_name: str = ""
        self.task_id: str = ""
        self.version: str = ""
        self.actor_num: int = -1
        self.actor_idx: int = 0
        self.max_train_count: Optional[int] = None
        self.timeout: Optional[int] = None
        self.create_time: Optional[datetime.datetime] = None

        self.is_create_env: Optional[bool] = None

        if self.uid == "":
            self.uid = str(uuid.uuid4()) if self.uid == "" else self.uid
            logger.info(f"uid: {self.uid}")


class TaskManager:
    def __init__(
        self,
        redis_params: RedisParameters,
        role: str = "other",
        keepalive_interval: int = 10,
        keepalive_threshold: int = 101,
        uid: str = "",
        used_device_tf: str = "/CPU",
        used_device_torch: str = "cpu",
    ):
        self._config = None

        if redis_params.url == "NO_USE":
            return
        self._connector = redis_params.create_connector()

        self.task_name = redis_params.task_name
        self.params = TaskManagerParams(
            role,
            keepalive_interval,
            keepalive_threshold,
            uid,
            used_device_tf,
            used_device_torch,
        )
        self.params.task_name = self.task_name

    @staticmethod
    def new_connector(connector: RedisConnector, task_manager_params: TaskManagerParams):
        t = TaskManager(RedisParameters(url="NO_USE"))
        t._connector = connector
        t.params = task_manager_params
        t.task_name = task_manager_params.task_name
        return t

    def get_now_str(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S %z")

    def create(
        self,
        task_config: TaskConfig,
        parameter: Any,
        uid: str = "",
    ) -> None:
        assert (
            task_config.context.max_train_count > 0 or task_config.context.timeout > 0
        ), "Please specify 'max_train_count' or 'timeout'."
        self._connector.set(f"{self.task_name}:status", "END")  # first

        self.params.task_id = str(uuid.uuid4()) if uid == "" else uid
        self.params.version = srl.__version__
        self.config = task_config
        self.params.actor_num = task_config.context.actor_num
        self.params.max_train_count = task_config.context.max_train_count
        self.params.timeout = task_config.context.timeout
        self.params.create_time = datetime.datetime.now(datetime.timezone.utc)

        self._connector.set(f"{self.task_name}:id", self.params.task_id)
        self._connector.set(f"{self.task_name}:version", self.params.version)
        self._connector.set(f"{self.task_name}:raw:config", pickle.dumps(task_config))
        self._connector.set(f"{self.task_name}:raw:parameter", pickle.dumps(parameter))
        self._connector.set(f"{self.task_name}:actor_num", str(self.params.actor_num))
        self._connector.set(f"{self.task_name}:max_train_count", str(self.params.max_train_count))
        self._connector.set(f"{self.task_name}:timeout", str(self.params.timeout))
        self._connector.set(f"{self.task_name}:create_time", self.params.create_time.strftime("%Y-%m-%d %H:%M:%S %z"))
        # --- val
        self._connector.set(f"{self.task_name}:setup_memory", "0")
        self._connector.set(f"{self.task_name}:train_count", "0")
        for k in self._connector.get_keys(f"{self.task_name}:trainer:*"):
            self._connector.delete(k)
        self._connector.set(f"{self.task_name}:trainer:id", "")
        for k in self._connector.get_keys(f"{self.task_name}:actor:*"):
            self._connector.delete(k)
        for i in range(task_config.context.actor_num):
            self._connector.set(f"{self.task_name}:actor:{i}:id", "")

        self._connector.set(f"{self.task_name}:status", "ACTIVE")  # last
        logger.info("create new task")
        self.add_log("Create new task")

    def get_task_id(self) -> str:
        if self.params.task_id == "":
            body = self._connector.get(f"{self.task_name}:id")
            self.params.task_id = "" if body is None else body.decode()
        return self.params.task_id

    def get_version(self) -> str:
        if self.params.version == "":
            body = self._connector.get(f"{self.task_name}:version")
            self.params.version = "" if body is None else body.decode()
        return self.params.version

    def check_version(self):
        try:
            v = self.get_version()
            if v == "":
                return
            if not compare_equal_version(v, srl.__version__):
                logger.warning(f"SRL version is different({v} != {srl.__version__})")
        except Exception as e:
            logger.info(f"SRL version check fail: {e}")

    def get_config(self) -> Optional[TaskConfig]:
        if self._config is not None:
            return self._config
        body = self._connector.get(f"{self.task_name}:raw:config")
        if body is None:
            return None
        self._config = cast(TaskConfig, pickle.loads(body))

        # device
        self._config.context.create_controller().set_device(
            self.params.used_device_tf,
            self.params.used_device_torch,
        )
        self._config.context.used_device_tf = self.params.used_device_tf
        self._config.context.used_device_torch = self.params.used_device_torch

        # local
        self.params.actor_num = self._config.context.actor_num
        self.params.max_train_count = self._config.context.max_train_count
        self.params.timeout = self._config.context.timeout

        return self._config

    def get_actor_num(self) -> int:
        if self.params.actor_num < 0:
            body = self._connector.get(f"{self.task_name}:actor_num")
            self.params.actor_num = -1 if body is None else int(body.decode())
        return self.params.actor_num

    def get_max_train_count(self) -> int:
        if self.params.max_train_count is None:
            body = self._connector.get(f"{self.task_name}:max_train_count")
            self.params.max_train_count = -1 if body is None else int(body.decode())
        return self.params.max_train_count

    def get_timeout(self) -> int:
        if self.params.timeout is None:
            body = self._connector.get(f"{self.task_name}:timeout")
            self.params.timeout = -1 if body is None else int(body.decode())
        return self.params.timeout

    def get_create_time(self) -> datetime.datetime:
        if self.params.create_time is None:
            body = self._connector.get(f"{self.task_name}:create_time")
            if body is None:
                return datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc)
            self.params.create_time = datetime.datetime.strptime(body.decode(), "%Y-%m-%d %H:%M:%S %z")
        return self.params.create_time

    def get_train_count(self) -> int:
        body = self._connector.get(f"{self.task_name}:train_count")
        return 0 if body is None else int(body.decode())

    def set_train_count(self, start_train_count: int, train_count: int):
        self._connector.set(f"{self.task_name}:train_count", str(start_train_count + train_count))

    # --- task trainer
    def get_trainer(self, key: str) -> str:
        body = self._connector.get(f"{self.task_name}:trainer:{key}")
        return "" if body is None else body.decode()

    def set_trainer(self, key: str, value: str) -> None:
        self._connector.set(f"{self.task_name}:trainer:{key}", value)

    def get_trainer_update_time(self) -> datetime.datetime:
        body = self._connector.get(f"{self.task_name}:trainer:update_time")
        if body is None:
            return datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc)
        return datetime.datetime.strptime(body.decode(), "%Y-%m-%d %H:%M:%S %z")

    # --- task actor
    def get_actor(self, idx: int, key: str) -> str:
        body = self._connector.get(f"{self.task_name}:actor:{idx}:{key}")
        return "" if body is None else body.decode()

    def set_actor(self, idx: int, key: str, value: str) -> None:
        self._connector.set(f"{self.task_name}:actor:{idx}:{key}", value)

    def get_actor_update_time(self, idx: int) -> datetime.datetime:
        body = self._connector.get(f"{self.task_name}:actor:{idx}:update_time")
        if body is None:
            return datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc)
        return datetime.datetime.strptime(body.decode(), "%Y-%m-%d %H:%M:%S %z")

    # --- status
    def get_status(self) -> str:
        body = self._connector.get(f"{self.task_name}:status")
        return "" if body is None else body.decode()

    def set_status(self, status: str):
        self._connector.set(f"{self.task_name}:status", status)

    def is_finished(self) -> bool:
        if self.get_status() == "END":
            return True

        # end check
        if self.get_max_train_count() > 0 and self.get_train_count() >= self.get_max_train_count():
            self.finished("train count over")
            return True

        if self.get_timeout() > 0:
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            if (now_utc - self.get_create_time()).total_seconds() >= self.get_timeout():
                self.finished("timeout")
                return True

        return False

    def finished(self, reason: str = "call"):
        self.set_status("END")
        self._connector.set(f"{self.task_name}:setup_memory", "0")
        s = f"Task finished({reason})"
        self.add_log(s)
        logger.info(s)

    # --- setup memory
    def setup_memory(self, memory: IMemoryReceiver, is_purge: bool) -> None:
        if is_purge and not self.is_setup_memory():
            memory.memory_purge()
            logger.info("memory purge")
            self.add_log("memory purge")

        for idx in range(self.get_actor_num()):
            self.set_actor(idx, "q_send_count", "0")
        self.set_trainer("q_recv_count", "0")
        self._connector.set(f"{self.task_name}:setup_memory", "1")

    def is_setup_memory(self) -> bool:
        body = self._connector.get(f"{self.task_name}:setup_memory")
        if body is None:
            return False
        return body.decode() == "1"

    # --- log
    def add_log(self, msg: str):
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y/%m/%d %H:%M:%S")
        s = f"{now} {self.params.uid} [{self.params.role}] {msg}"
        self._connector.rpush(f"{self.task_name}:logs", s)
        logger.info(s)

    # --- runner
    def is_create_env(self) -> bool:
        if self.params.is_create_env is None:
            task_config = self.get_config()
            if task_config is None:
                return False
            runner = srl.Runner(
                task_config.context.env_config,
                task_config.context.rl_config,
                task_config.config,
                task_config.context,
            )

            try:
                runner.make_env()
                self.params.is_create_env = True
            except Exception as e:
                logger.warning(f"Env load fail: {e}")
                self.params.is_create_env = False

        return self.params.is_create_env

    def create_runner(self, read_parameter: bool = True) -> Optional[srl.Runner]:
        task_config = self.get_config()
        if task_config is None:
            return None
        runner = srl.Runner(
            task_config.context.env_config,
            task_config.context.rl_config,
            task_config.config,
            task_config.context,
        )
        if read_parameter:
            params = self._connector.parameter_read()
            if params is not None:
                runner.make_parameter().restore(params)
        return runner
