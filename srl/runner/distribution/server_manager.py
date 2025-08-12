import datetime
import logging
import pickle
import random
import time
import traceback
import uuid
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Union, cast

import srl
from srl.base.context import RunContext
from srl.base.exception import DistributionError
from srl.base.rl.parameter import RLParameter
from srl.runner.distribution.connector_configs import (
    IMemoryReceiver,
    IMemorySender,
    IMemoryServerParameters,
    RedisParameters,
)
from srl.utils.common import compare_equal_version

if TYPE_CHECKING:
    from srl.runner.distribution.callback import DistributionCallback
    from srl.runner.distribution.connectors.redis_ import RedisConnector

logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    context: RunContext
    queue_capacity: int = 1000
    trainer_parameter_send_interval: int = 1  # sec
    actor_parameter_sync_interval: int = 1  # sec


class ServerManager:
    def __init__(self, redis_params: RedisParameters, memory_params: Optional[IMemoryServerParameters]):
        self.redis_params = redis_params
        self.memory_params = memory_params

        self._task_manager = None
        self._redis_connector = None
        self._memory_receiver = None
        self._memory_sender = None

    def get_redis_connector(self) -> "RedisConnector":
        if self._redis_connector is None:
            self._redis_connector = self.redis_params.create()
        return self._redis_connector

    def get_task_manager(self, role: Literal["other", "actor", "trainer"], uid: str = "") -> "TaskManager":
        if self._task_manager is None:
            self._task_manager = TaskManager(self.get_redis_connector(), role, uid)
        return self._task_manager

    def get_parameter_server(self) -> "RedisConnector":
        return self.get_redis_connector()

    def get_memory_receiver(self) -> IMemoryReceiver:
        if self.memory_params is None:
            return self.get_redis_connector()
        if self._memory_receiver is None:
            self._memory_receiver = self.memory_params.create_memory_receiver()
        return self._memory_receiver

    def get_memory_sender(self) -> IMemorySender:
        if self.memory_params is None:
            return self.get_redis_connector()
        if self._memory_sender is None:
            self._memory_sender = self.memory_params.create_memory_sender()
        return self._memory_sender


class TaskManager:
    def __init__(
        self,
        redis: Union["RedisConnector", RedisParameters],
        role: Literal["other", "actor", "trainer", "client"] = "other",
        uid: str = "",
    ):
        if isinstance(redis, RedisParameters):
            self._board = redis.create()
        else:
            self._board = redis
        self.role = role
        self.uid = str(uuid.uuid4()) if uid == "" else uid

        self._task_id = ""
        self._version: str = ""
        self._actor_num: int = -1
        self._max_train_count: Optional[int] = None
        self._timeout: Optional[float] = None
        self._create_time: Optional[datetime.datetime] = None
        self._config: Optional[TaskConfig] = None

    def create_task(self, task_config: TaskConfig, parameter_dat: Optional[Any], uid: str = "") -> None:
        task_config.context.check_context_parameter()
        self._board.set("status", "INIT")  # first

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        self._task_id = str(uuid.uuid4()) if uid == "" else uid

        # --- init board
        for k, v in {
            "id": self._task_id,
            "version": srl.__version__,
            "actor_num": task_config.context.actor_num,
            "max_train_count": task_config.context.max_train_count,
            "timeout": task_config.context.timeout,
            "create_time": now_utc.strftime("%Y-%m-%d %H:%M:%S %z"),
            # --- val
            "train_count": "0",
            "setup_memory": "0",  # （初期化はmemoryへのconnect的にTrainerで実施）
        }.items():
            self._board.set(k, v)

        self.write_config(task_config)
        self.write_parameter_dat(parameter_dat, init=True)

        # --- trainer board
        for k in self._board.get_keys("trainer:*"):
            self._board.delete(k)
        self._board.set("trainer:id", "")
        self._board.set("trainer:q_recv_count", "0")

        # --- actor board
        for k in self._board.get_keys("actor:*"):
            self._board.delete(k)
        for i in range(task_config.context.actor_num):
            self._board.set(f"actor:{i}:id", "")
            self._board.set(f"actor:{i}:q_send_count", "0")

        self._board.set("status", "ACTIVE")  # last
        self.log("Create new task")

    def setup_memory(self, memory: IMemoryReceiver, is_purge: bool) -> None:
        if is_purge and not self.is_setup_memory():
            memory.memory_purge()
            self.log("memory purge")
        self._board.set("setup_memory", "1")

    def is_setup_memory(self) -> bool:
        body = self._board.get("setup_memory")
        if body is None:
            return False
        return body.decode() == "1"

    def assign(self, actor_idx: int = 0) -> bool:
        self.check_version()
        if self.role == "trainer":
            self.set_trainer("id", self.uid)
            self.set_trainer("update_time", self.get_now_str())
            self.log(f"Trainer assigned({self.uid})")
        elif self.role == "actor":
            self.set_actor(actor_idx, "id", self.uid)
            self.set_actor(actor_idx, "update_time", self.get_now_str())
            self.log(f"Actor{actor_idx} assigned({self.uid})")
        else:
            raise ValueError(self.role)

        return True

    def unassign(self, role: str, actor_idx: int = 0, reason: str = ""):
        if role == "trainer":
            self.set_trainer("id", "")
            self.set_trainer("update_time", self.get_now_str())
            self.log(f"Trainer unassigned: {reason}")
        elif role == "actor":
            self.set_actor(actor_idx, "id", "")
            self.set_actor(actor_idx, "update_time", self.get_now_str())
            self.log(f"Actor{actor_idx} unassigned: {reason}")
        else:
            raise ValueError(self.role)

    def keepalive_trainer(self, start_train_count: int, share_data: Any):
        # --- 2重にアサインされていないかチェック
        now_uid = self.get_trainer("id")
        if now_uid != self.uid:
            if not self.is_finished():
                # アサインされていたらランダム秒まって止める
                s = f"Another trainer has been assigned. my:{self.uid}, another: {now_uid}"
                self.log(s)
                time.sleep(random.randint(0, 5))
                raise DistributionError(s)

        self.set_trainer("q_recv_count", str(share_data.q_recv_count))
        self.set_train_count(start_train_count, share_data.train_count)
        self.set_trainer("train", str(share_data.train_count))
        self.set_trainer("update_time", self.get_now_str())

    def keepalive_actor(self, actor_id: int, step: int, q_send_count: int):
        # --- 2重にアサインされていないかチェック
        now_uid = self.get_actor(actor_id, "id")
        if now_uid != self.uid:
            # アサインされていたらランダム秒まって止める
            s = f"Another actor has been assigned. my:{self.uid}, another: {now_uid}"
            self.log(s)
            time.sleep(random.randint(0, 5))
            raise DistributionError(s)

        self.set_actor(actor_id, "step", str(step))
        self.set_actor(actor_id, "q_send_count", str(q_send_count))
        self.set_actor(actor_id, "update_time", self.get_now_str())

    # -----------------------------------------------

    def get_now_str(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S %z")

    def get_task_id(self) -> str:
        if self._task_id == "":
            body = self._board.get("id")
            self._task_id = "" if body is None else body.decode()
        return self._task_id

    def get_version(self) -> str:
        if self._version == "":
            body = self._board.get("version")
            self._version = "" if body is None else body.decode()
        return self._version

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
        self._board.set("raw:config", val)

    def get_config(self) -> Optional[TaskConfig]:
        if self._config is not None:
            return self._config
        body = self._board.get("raw:config")
        if body is None:
            return None
        body = pickle.loads(zlib.decompress(body))
        self._config = cast(TaskConfig, body)
        return self._config

    def get_actor_num(self) -> int:
        if self._actor_num < 0:
            body = self._board.get("actor_num")
            self._actor_num = -1 if body is None else int(body.decode())
        return self._actor_num

    def get_max_train_count(self) -> int:
        if self._max_train_count is None:
            body = self._board.get("max_train_count")
            self._max_train_count = -1 if body is None else int(body.decode())
        return self._max_train_count

    def get_timeout(self) -> float:
        if self._timeout is None:
            body = self._board.get("timeout")
            self._timeout = -1 if body is None else int(body.decode())
        return self._timeout

    def get_create_time(self) -> datetime.datetime:
        if self._create_time is None:
            body = self._board.get("create_time")
            if body is None:
                return datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc)
            self._create_time = datetime.datetime.strptime(body.decode(), "%Y-%m-%d %H:%M:%S %z")
        return self._create_time

    def get_train_count(self) -> int:
        body = self._board.get("train_count")
        return 0 if body is None else int(body.decode())

    def set_train_count(self, start_train_count: int, train_count: int):
        self._board.set("train_count", str(start_train_count + train_count))

    # --- task trainer
    def get_trainer(self, key: str) -> str:
        body = self._board.get(f"trainer:{key}")
        return "" if body is None else body.decode()

    def set_trainer(self, key: str, value: str) -> None:
        self._board.set(f"trainer:{key}", value)

    def get_trainer_update_time(self) -> datetime.datetime:
        body = self._board.get("trainer:update_time")
        if body is None:
            return datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc)
        return datetime.datetime.strptime(body.decode(), "%Y-%m-%d %H:%M:%S %z")

    # --- task actor
    def get_actor(self, idx: int, key: str) -> str:
        body = self._board.get(f"actor:{idx}:{key}")
        return "" if body is None else body.decode()

    def set_actor(self, idx: int, key: str, value: str) -> None:
        self._board.set(f"actor:{idx}:{key}", value)

    def get_actor_update_time(self, idx: int) -> datetime.datetime:
        body = self._board.get(f"actor:{idx}:update_time")
        if body is None:
            return datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc)
        return datetime.datetime.strptime(body.decode(), "%Y-%m-%d %H:%M:%S %z")

    # --- status
    def get_status(self) -> str:
        body = self._board.get("status")
        return "" if body is None else body.decode()

    def set_status(self, status: str):
        self._board.set("status", status)

    def is_finished(self) -> bool:
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
        self.log(s)
        logger.info(s)

    # --- log
    def log(self, msg: str):
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y/%m/%d %H:%M:%S")
        s = f"{now} {self.uid} [{self.role}] {msg}"
        self._board.rpush("logs", s)
        logger.info(s)

    # ---------------------------------
    # runner
    # ---------------------------------
    def write_parameter_dat(self, parameter_dat: Optional[Any], init: bool = False):
        self._board.parameter_write(parameter_dat, init=init)

    def write_parameter(self, parameter: RLParameter, init: bool = False):
        self._board.parameter_write(parameter.backup(serialized=True), init=init)

    def read_parameter(self, parameter: RLParameter) -> bool:
        params = self._board.parameter_read()
        if params is None:
            return False
        parameter.restore(params, from_serialized=True)
        return True

    def create_runner(self, read_parameter: bool = True) -> Optional[srl.Runner]:
        task_config = self.get_config()
        if task_config is None:
            return None

        runner = srl.Runner(context=task_config.context)
        if read_parameter:
            self.read_parameter(runner.make_parameter())
        return runner

    def create_parameter(self) -> Optional[RLParameter]:
        task_config = self.get_config()
        if task_config is None:
            return None
        parameter = task_config.context.rl_config.make_parameter()
        self.read_parameter(parameter)
        return parameter

    # ---------------------------------
    # facade
    # ---------------------------------
    def train_wait(
        self,
        # --- progress
        enable_progress: bool = True,
        progress_interval: int = 60 * 1,
        # --- other
        progress_kwargs: dict = {},
        checkpoint_kwargs: Optional[dict] = None,
        history_on_file_kwargs: Optional[dict] = None,
        callbacks: List["DistributionCallback"] = [],
        raise_exception: bool = True,
    ):
        callbacks = callbacks[:]

        if enable_progress:
            from srl.runner.distribution.callbacks.print_progress import PrintProgress

            callbacks.append(
                PrintProgress(
                    interval=progress_interval,
                    enable_eval=progress_kwargs.get("enable_eval", False),
                    eval_episode=progress_kwargs.get("eval_episode", 1),
                    eval_timeout=progress_kwargs.get("eval_timeout", -1),
                    eval_max_steps=progress_kwargs.get("eval_max_steps", -1),
                    eval_players=progress_kwargs.get("eval_players", []),
                    eval_shuffle_player=progress_kwargs.get("eval_shuffle_player", False),
                )
            )
            logger.info("add callback PrintProgress")

        if checkpoint_kwargs is not None:
            from srl.runner.distribution.callbacks.checkpoint import Checkpoint

            callbacks.append(Checkpoint(**checkpoint_kwargs))
            logger.info(f"add callback Checkpoint: {checkpoint_kwargs['save_dir']}")

        if history_on_file_kwargs is not None:
            from srl.runner.distribution.callbacks.history_on_file import HistoryOnFile

            callbacks.append(HistoryOnFile(**history_on_file_kwargs))
            logger.info(f"add callback Checkpoint: {history_on_file_kwargs['save_dir']}")

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
