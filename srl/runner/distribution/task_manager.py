import datetime
import logging
import pickle
import time
import traceback
import uuid
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union, cast

import srl
from srl.base.rl.parameter import RLParameter
from srl.base.run.context import RLWorkerType, StrWorkerType
from srl.runner.distribution.connectors.parameters import RedisParameters
from srl.runner.distribution.connectors.redis_ import RedisConnector
from srl.runner.distribution.interface import IMemoryReceiver
from srl.runner.runner import TaskConfig
from srl.utils.common import compare_equal_version

if TYPE_CHECKING:
    from srl.runner.distribution.callback import DistributionCallback

logger = logging.getLogger(__name__)


@dataclass
class TaskManagerParams:
    role: str
    keepalive_interval: int = 10
    keepalive_threshold: int = 101
    uid: str = ""
    framework: str = ""
    used_device_tf: str = "/CPU"
    used_device_torch: str = "cpu"

    def __post_init__(self):
        self.init()

    def init(self):
        self.task_name: str = ""
        self.task_id: str = ""
        self.version: str = ""
        self.actor_num: int = -1
        self.max_train_count: Optional[int] = None
        self.timeout: Optional[float] = None
        self.create_time: Optional[datetime.datetime] = None

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
        framework: str = "",
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
            framework,
            used_device_tf,
            used_device_torch,
        )
        self.params.task_name = self.task_name

    def reset(self):
        self._config = None
        self.params.init()
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

    def create_task(
        self,
        task_config: TaskConfig,
        parameter: RLParameter,
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
        self.write_config(task_config)
        self.write_parameter(parameter)
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
        # 別タスクかどうかチェックする場合も使う
        body = self._connector.get(f"{self.task_name}:id")
        task_id = "" if body is None else body.decode()
        if self.params.task_id == "":
            self.params.task_id = task_id
        return task_id

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

    def write_config(self, config: TaskConfig):
        val = zlib.compress(pickle.dumps(config))
        self._connector.set(f"{self.task_name}:raw:config", val)

    def get_config(self) -> Optional[TaskConfig]:
        if self._config is not None:
            return self._config
        body = self._connector.get(f"{self.task_name}:raw:config")
        if body is None:
            return None
        body = pickle.loads(zlib.decompress(body))
        self._config = cast(TaskConfig, body)

        # device
        self._config.context.set_device(
            self.params.framework,
            self.params.used_device_tf,
            self.params.used_device_torch,
        )

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

    def get_timeout(self) -> float:
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
        if self.params.task_id != "" and self.params.task_id != self.get_task_id():
            logger.info(f"is_finished: different ID({self.params.task_id} != {self.get_task_id()})")
            return True

        if self.get_status() == "END":
            logger.info("is_finished: status END")
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

    # ---------------------------------
    # runner
    # ---------------------------------
    def write_parameter(self, parameter: RLParameter):
        params = parameter.backup(to_cpu=True)
        if params is None:
            return
        self._connector.parameter_update(params)

    def read_parameter(self, parameter: RLParameter) -> bool:
        params = self._connector.parameter_read()
        if params is None:
            return False
        parameter.restore(params, from_cpu=True)
        return True

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
            self.read_parameter(runner.make_parameter())
        return runner

    # ---------------------------------
    # facade
    # ---------------------------------
    def train_wait(
        self,
        # --- progress
        enable_progress: bool = True,
        progress_interval: int = 60 * 1,
        # --- checkpoint
        checkpoint_save_dir: str = "",
        checkpoint_interval: int = 60 * 20,
        # --- history
        history_save_dir: str = "",
        history_interval: int = 10,
        history_add_history: bool = True,
        # --- eval
        enable_eval: bool = True,
        eval_episode: int = 1,
        eval_timeout: float = -1,
        eval_max_steps: int = -1,
        eval_players: List[Union[None, StrWorkerType, RLWorkerType]] = [],
        eval_shuffle_player: bool = False,
        # --- other
        callbacks: List["DistributionCallback"] = [],
        raise_exception: bool = True,
    ):
        callbacks = callbacks[:]

        if enable_progress:
            from srl.runner.distribution.callbacks.print_progress import PrintProgress

            callbacks.append(
                PrintProgress(
                    interval=progress_interval,
                    enable_eval=enable_eval,
                    eval_episode=eval_episode,
                    eval_timeout=eval_timeout,
                    eval_max_steps=eval_max_steps,
                    eval_players=eval_players,
                    eval_shuffle_player=eval_shuffle_player,
                )
            )
            logger.info("add callback PrintProgress")

        if checkpoint_save_dir != "":
            from srl.runner.distribution.callbacks.checkpoint import Checkpoint

            callbacks.append(
                Checkpoint(
                    save_dir=checkpoint_save_dir,
                    interval=checkpoint_interval,
                    enable_eval=enable_eval,
                    eval_episode=eval_episode,
                    eval_timeout=eval_timeout,
                    eval_max_steps=eval_max_steps,
                    eval_players=eval_players,
                    eval_shuffle_player=eval_shuffle_player,
                )
            )
            logger.info(f"add callback Checkpoint: {checkpoint_save_dir}")

        if history_save_dir != "":
            from srl.runner.distribution.callbacks.history_on_file import HistoryOnFile

            callbacks.append(
                HistoryOnFile(
                    save_dir=history_save_dir,
                    interval=history_interval,
                    add_history=history_add_history,
                    enable_eval=enable_eval,
                    eval_episode=eval_episode,
                    eval_timeout=eval_timeout,
                    eval_max_steps=eval_max_steps,
                    eval_players=eval_players,
                    eval_shuffle_player=eval_shuffle_player,
                )
            )
            logger.info(f"add callback Checkpoint: {checkpoint_save_dir}")

        # callbacks
        [c.on_start(self) for c in callbacks]

        if raise_exception:
            while True:
                time.sleep(1)

                if self.is_finished():
                    break

                _stop_flags = [c.on_polling(self) for c in callbacks]
                if True in _stop_flags:
                    break
        else:
            while True:
                try:
                    time.sleep(1)

                    if self.is_finished():
                        break

                    _stop_flags = [c.on_polling(self) for c in callbacks]
                    if True in _stop_flags:
                        break
                except Exception:
                    logger.error(traceback.format_exc())

        [c.on_end(self) for c in callbacks]
