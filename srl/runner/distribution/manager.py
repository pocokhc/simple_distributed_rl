import datetime
import logging
import pickle
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import pika
import pika.exceptions
import redis

from srl.runner.runner import TaskConfig

logger = logging.getLogger(__name__)


class IMemoryConnector(ABC):
    @abstractmethod
    def ping(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def memory_add(self, task_id: str, dat: Any) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def memory_recv(self, task_id: str) -> Optional[Any]:
        raise NotImplementedError()

    @abstractmethod
    def memory_size(self, task_id: str) -> int:
        raise NotImplementedError()

    @abstractmethod
    def memory_delete(self, task_id: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def memory_exist(self, task_id: str) -> bool:
        raise NotImplementedError()


@dataclass
class ServerParameters:
    redis_url: str = ""
    redis_host: str = ""
    redis_kwargs: dict = field(default_factory=dict)
    rabbitmq_url: str = ""
    rabbitmq_host: str = ""
    rabbitmq_port: int = 5672
    rabbitmq_username: str = "guest"
    rabbitmq_password: str = "guest"
    rabbitmq_virtual_host: str = "/"
    rabbitmq_ssl: bool = False
    rabbitmq_kwargs: dict = field(default_factory=dict)
    keepalive_interval: int = 10


class RabbitMQConnector(IMemoryConnector):
    def __init__(self, parameter: ServerParameters):
        if parameter.rabbitmq_url != "":
            self._pika_params = pika.URLParameters(parameter.rabbitmq_url)
        else:
            ssl_options = None
            if parameter.rabbitmq_ssl:
                import ssl

                ssl_options = pika.SSLOptions(ssl.create_default_context())
            self._pika_params = pika.ConnectionParameters(
                parameter.rabbitmq_host,
                parameter.rabbitmq_port,
                parameter.rabbitmq_virtual_host,
                ssl_options=ssl_options,
                credentials=pika.PlainCredentials(
                    parameter.rabbitmq_username,
                    parameter.rabbitmq_password,
                ),
                **parameter.rabbitmq_kwargs,
            )

        self.connection = None
        self.channel = None
        self.is_queue: Dict[str, bool] = {}

    def close(self):
        if self.connection is not None:
            try:
                self.connection.close()
                logger.debug("Connection to RabbitMQ closed")
            except pika.exceptions.AMQPError as e:
                logger.error(f"RabbitMQ close error: {str(e)}")

        self.connection = None
        self.channel = None
        self.is_queue = {}

    def connect(self) -> bool:
        if self.connection is not None:
            return True
        try:
            self.connection = pika.BlockingConnection(self._pika_params)
            self.channel = self.connection.channel()
            logger.debug("Connected to RabbitMQ")
            return True
        except pika.exceptions.AMQPError:
            logger.error(traceback.format_exc())
            self.connection = None
            self.channel = None
        return False

    def ping(self) -> bool:
        return self.connect()

    def memory_add(self, task_id: str, dat: Any) -> bool:
        try:
            if not self.channel:
                self.connect()
                assert self.channel is not None
            q_name = f"task:{task_id}"
            if q_name not in self.is_queue:
                self.channel.queue_declare(queue=q_name)
                self.is_queue[q_name] = True
            dat = pickle.dumps(dat)
            self.channel.basic_publish(exchange="", routing_key=q_name, body=dat)
            return True
        except pika.exceptions.AMQPError:
            logger.error(traceback.format_exc())
            self.close()
        return False

    def memory_recv(self, task_id: str) -> Optional[Any]:
        try:
            if not self.channel:
                self.connect()
                assert self.channel is not None
            q_name = f"task:{task_id}"
            if q_name not in self.is_queue:
                self.channel.queue_declare(queue=q_name)
                self.is_queue[q_name] = True
            _, _, body = self.channel.basic_get(queue=q_name, auto_ack=True)
            return body if body is None else pickle.loads(cast(bytes, body))
        except pika.exceptions.AMQPError:
            logger.error(traceback.format_exc())
            self.close()
        return None

    def memory_size(self, task_id: str) -> int:
        try:
            if not self.channel:
                self.connect()
                assert self.channel is not None
            q_name = f"task:{task_id}"
            if q_name not in self.is_queue:
                self.channel.queue_declare(queue=q_name)
                self.is_queue[q_name] = True
            queue_info = self.channel.queue_declare(queue=q_name, passive=True)
            return queue_info.method.message_count
        except pika.exceptions.ChannelClosed as e:
            self.close()
            self.connect()
            if e.reply_code == 404:  # NOT_FOUND
                return 0
        except Exception:
            logger.error(traceback.format_exc())
            self.close()
        return 0

    def memory_delete(self, task_id: str) -> None:
        if not self.channel:
            self.connect()
            assert self.channel is not None
        self.channel.queue_delete(f"task:{task_id}")

    def memory_exist(self, task_id: str) -> bool:
        if not self.channel:
            self.connect()
            assert self.channel is not None
        try:
            self.channel.queue_declare(f"task:{task_id}", passive=True)
            return True
        except pika.exceptions.ChannelClosed as e:
            self.close()
            if e.reply_code == 404:  # NOT_FOUND
                return False
        return False


class DistributedManager(IMemoryConnector):
    def __init__(self, parameter: ServerParameters):
        self.parameter = parameter

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
            self.parameter,
            self.uid,
            self.role,
            self.actor_idx,
        )

    @staticmethod
    def create(
        parameter,
        uid,
        role,
        actor_idx,
    ):
        m = DistributedManager(parameter)
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

    def connect(self) -> bool:
        if self.server is not None:
            return True
        try:
            if self.parameter.redis_url != "":
                self.server = redis.from_url(self.parameter.redis_url, **self.parameter.redis_kwargs)
                return True
            self.server = redis.Redis(self.parameter.redis_host, **self.parameter.redis_kwargs)
            return True
        except Exception:
            logger.error(traceback.format_exc())
            self.server = None
        return False

    def ping(self) -> bool:
        if self.connect():
            assert self.server is not None
            return cast(bool, self.server.ping())
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

    def create_memory_connector(self) -> IMemoryConnector:
        if self.parameter.rabbitmq_url == "" and self.parameter.rabbitmq_host == "":
            logger.info("memory use Redis")
            return self
        try:
            mq = RabbitMQConnector(self.parameter)
            mq.connect()
            logger.info("memory use RabbitMQ")
            return mq
        except Exception:
            logger.info("memory use Redis")
            return self

    # -----------------------------
    # memory
    # -----------------------------
    def memory_add(self, task_id: str, dat: Any) -> bool:
        try:
            if not self.connect():
                return False
            assert self.server is not None
            dat = pickle.dumps(dat)
            self.server.rpush(f"task:{task_id}:memory", dat)
            return True
        except Exception:
            logger.error(traceback.format_exc())
            self.close()
        return False

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
                return 0
            assert self.server is not None
            return cast(int, self.server.llen(f"task:{task_id}:memory"))
        except Exception:
            logger.error(traceback.format_exc())
            self.close()
        return 0

    def memory_delete(self, task_id: str) -> None:
        self.connect()
        assert self.server is not None
        self.server.delete(f"task:{task_id}:memory")

    def memory_exist(self, task_id: str) -> bool:
        return self.server_exists(f"task:{task_id}:memory")

    # -----------------------------
    # parameter
    # -----------------------------
    def parameter_read(self, task_id: str = ""):
        if task_id == "":
            task_id = self.uid
        params = self.server_get(f"task:{task_id}:parameter")
        return params if params is None else pickle.loads(params)

    def parameter_update(self, task_id: str, parameter: Any) -> bool:
        # 自分のidが登録されている場合のみ更新
        _tid = self.task_get_trainer(task_id, "id")
        assert _tid == self.uid
        self.server_set(f"task:{task_id}:parameter", pickle.dumps(parameter))
        return True

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

        self.task_log(task_id, "create task")
        return task_id

    def task_end(self, task_id: str):
        if not self.server_exists(f"taskid:{task_id}"):
            return
        self.server_set(f"task:{task_id}:status", "END")
        logger.info(f"task end: {task_id}")
        self.task_log(task_id, "End")

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

    def task_get_config(self, task_id: str) -> Optional[TaskConfig]:
        config = self.server_get(f"task:{task_id}:config")
        assert config is not None
        config = pickle.loads(config)
        return cast(TaskConfig, config)

    def task_get_ids(self) -> List[str]:
        if not self.connect():
            return []
        assert self.server is not None
        task_ids = []
        for key in self.server.scan_iter("taskid:*"):
            task_id = self.server_get(key)
            if task_id is not None:
                task_ids.append(task_id.decode())
        return task_ids

    def task_assign_by_my_id(self) -> Tuple[str, int]:
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

                self.server_set(f"task:{task_id}:trainer:id", self.uid)
                self.server_set(f"task:{task_id}:trainer:health", now_str)
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
                    self.server_set(f"task:{task_id}:actor:{i}:id", self.uid)
                    self.server_set(f"task:{task_id}:actor:{i}:health", now_str)
                    logger.info(f"Actor{i} assigned({self.uid}): {task_id}")
                    self.task_log(task_id, f"Actor{i} assigned")
                    return task_id, i

        return "", 0

    def _task_log_push(self, key: str, value: str):
        if not self.connect():
            return False
        assert self.server is not None
        self.server.rpush(key, value)

    def task_log(self, task_id: str, msg: str) -> None:
        if not self.server_exists(f"taskid:{task_id}"):
            return
        now = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        msg = f"{now} [{self.role}] {msg} ({self.uid})"
        self._task_log_push(f"task:{task_id}:logs", msg)

    def keepalive(self, task_id: str, do_now: bool = False) -> bool:
        if not self.server_exists(f"taskid:{task_id}"):
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
        self.connect()
        assert self.server is not None
        memory_manager = self.create_memory_connector()
        for id in self.task_get_ids():
            status = self.task_get_status(id)
            if status != "END":
                continue

            # --- delete queue
            memory_manager.memory_delete(id)

            # --- delete key
            keys = cast(list, self.server.keys(match=f"task:{id}:*"))
            self.server.delete(*keys)

            # すべて消えてたらidも消す
            keys = cast(list, self.server.keys(match=f"task:{id}:*"))
            if not memory_manager.memory_exist(id) and len(keys) == 0:
                self.server.delete(f"taskid:{id}")
                logger.info(f"task deleted: {id}")
