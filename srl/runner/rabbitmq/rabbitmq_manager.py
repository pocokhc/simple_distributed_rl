import datetime
import json
import logging
import pickle
import time
import traceback
import uuid
from typing import Any, List, Optional, Tuple

import pika
import pika.exceptions

logger = logging.getLogger(__name__)


class RabbitMQManager:
    def __init__(
        self,
        host: str,
        port: int = 5672,
        username: str = "guest",
        password: str = "guest",
        virtual_host: str = "/",
        wait_time: int = 5,
        keepalive_interval: int = 10,
        keepalive_limit_time: int = 60 * 10,
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.virtual_host = virtual_host
        self.uid = ""
        self.role = ""

        self.connection = None
        self.channel = None
        self.wait_time = wait_time
        self.keepalive_interval = keepalive_interval
        self.keepalive_limit_time = keepalive_limit_time

        self.queue_dict = {}
        self.user_queue_name_list = [
            "health",
            "board",
            "parameter",
            "last_parameter",
            "memory",
        ]
        self._keepalive_t0 = time.time()

        self._pika_params = pika.ConnectionParameters(
            self.host,
            self.port,
            self.virtual_host,
            credentials=pika.PlainCredentials(self.username, self.password),
        )
        logger.info("Start connecting to RabbitMQ")
        self.connect_loop()

    def copy_args(self):
        return (
            self.host,
            self.port,
            self.username,
            self.password,
            self.virtual_host,
            self.uid,
            self.role,
            self.wait_time,
            self.keepalive_interval,
            self.keepalive_limit_time,
        )

    @staticmethod
    def create(
        host,
        port,
        username,
        password,
        virtual_host,
        uid,
        role,
        wait_time,
        keepalive_interval,
        keepalive_limit_time,
    ):
        mq = RabbitMQManager(
            host,
            port,
            username,
            password,
            virtual_host,
            wait_time,
            keepalive_interval,
            keepalive_limit_time,
        )
        mq.uid = uid
        mq.role = role
        return mq

    def __del__(self):
        self.close()

    def close(self):
        if self.connection is not None and self.connection.is_open:
            try:
                self.connection.close()
                logger.info("Connection to RabbitMQ closed")
            except Exception as e:
                logger.error(f"RabbitMQ close error: {str(e)}")

        self.connection = None
        self.channel = None

    def connect_once(self) -> bool:
        if self.connection is not None and self.connection.is_open:
            return True
        try:
            self.connection = pika.BlockingConnection(self._pika_params)
            self.channel = self.connection.channel()
            logger.info("Connected to RabbitMQ")
            return True
        except Exception as e:
            logger.error(f"Error connecting to RabbitMQ: {str(e)}")
        return False

    def connect_loop(self) -> None:
        if self.connection is not None and self.connection.is_open:
            return
        retry = 0
        while True:
            try:
                self.connection = pika.BlockingConnection(self._pika_params)
                self.channel = self.connection.channel()
                logger.info("Connected to RabbitMQ")
                break
            except Exception as e:
                retry += 1
                logger.info(traceback.format_exc())
                logger.error(f"Error connecting to RabbitMQ: {str(e)}, retry {retry}")
                time.sleep(self.wait_time)

    # -----------------------------
    # queue
    # -----------------------------
    def create_queue_loop(self, queue: str) -> None:
        retry = 0
        while True:
            try:
                self.connect_loop()
                assert self.channel is not None
                self.channel.queue_declare(queue=queue)
                logger.info(f"Queue created: {queue}")
                return
            except Exception as e:
                retry += 1
                logger.error(f"create queue error: {str(e)}, retry {retry}")
                self.close()
            time.sleep(self.wait_time)

    def create_queue_once_if_not_exists(self, queue: str) -> None:
        try:
            if self.exist_queue_once(queue):
                return
            if not self.connect_once():
                return
            assert self.channel is not None
            self.channel.queue_declare(queue=queue)
            logger.info(f"Queue created: {queue}")
            return
        except Exception as e:
            logger.error(f"recv error: {str(e)}")
            self.close()

    def create_fanout_queue_once_if_not_exists(self, queue: str, exchange: str) -> None:
        try:
            if self.exist_queue_once(queue):
                return
            if not self.connect_once():
                return
            assert self.channel is not None
            self.channel.exchange_declare(exchange=exchange, exchange_type="fanout")
            self.channel.queue_declare(queue=queue)
            self.channel.queue_bind(exchange=exchange, queue=queue)
            logger.info(f"Queue(fanout) created: {queue}({exchange})")
            return
        except Exception as e:
            logger.error(f"recv error: {str(e)}")
            self.close()

    def purge_queue_once(self, queue: str) -> bool:
        try:
            if not self.connect_once():
                return False
            assert self.channel is not None
            self.channel.queue_purge(queue)
            logger.debug(f"Queue purged: {queue}")
            return True
        except Exception as e:
            logger.error(f"Error purge queue: {str(e)}")
            self.close()
        return False

    def purge_fanout_queue_loop(self, queue: str, exchange: str):
        retry = 0
        while True:
            try:
                assert self.channel is not None
                self.channel.queue_purge(queue)
                logger.debug(f"Queue purged: {queue}")
                return
            except Exception as e:
                retry += 1
                logger.error(f"Error purge queue: {str(e)}, retry {retry}")
                self.close()
                time.sleep(self.wait_time)
            self.connect_loop()
            self.create_fanout_queue_once_if_not_exists(queue, exchange)

    def exist_queue_once(self, queue: str) -> bool:
        try:
            if not self.connect_once():
                return False
            assert self.channel is not None
            self.channel.queue_declare(queue=queue, passive=True)
            return True
        except pika.exceptions.ChannelClosed as e:
            self.close()
            self.connect_once()
            if e.reply_code == 404:
                return False
            logger.error(f"Error checking the queue: {str(e)}")
        except Exception as e:
            logger.error(f"Error checking the queue: {str(e)}")
            self.close()
        return False

    def delete_queue_once(self, queue: str) -> bool:
        try:
            if not self.connect_once():
                return False
            assert self.channel is not None
            self.channel.queue_delete(queue)
            logger.info(f"queue deleted: {queue}")
            return True
        except Exception as e:
            logger.error(f"queue delete fail: {str(e)}")
        return False

    # -----------------------------
    # recv
    # -----------------------------
    def recv_once(self, queue: str, enable_pickle: bool = True) -> Optional[Any]:
        try:
            if not self.connect_once():
                return None
            assert self.channel is not None
            method_frame, header_frame, body = self.channel.basic_get(queue=queue, auto_ack=True)
            return pickle.loads(body) if enable_pickle and body is not None else body
        except Exception as e:
            logger.error(f"recv error: {str(e)}")
            self.close()
        return None

    def recv_loop(self, queue: str, enable_pickle: bool = True) -> Optional[Any]:
        retry = 0
        while True:
            try:
                self.connect_loop()
                assert self.channel is not None
                method_frame, header_frame, body = self.channel.basic_get(queue=queue, auto_ack=True)
                return pickle.loads(body) if enable_pickle and body is not None else body
            except Exception as e:
                retry += 1
                logger.error(f"recv error: {str(e)}, retry {retry}")
                self.close()
            time.sleep(self.wait_time)

    def recv_once_lastdata_and_purge(self, queue: str, enable_pickle: bool = True) -> Optional[Any]:
        try:
            if not self.connect_once():
                return None
            assert self.channel is not None
            queue_info = self.channel.queue_declare(queue=queue, passive=True)
            body = None
            for _ in range(queue_info.method.message_count):
                method_frame, header_frame, body = self.channel.basic_get(queue=queue, auto_ack=True)
            try:
                self.channel.queue_purge(queue)
            except Exception as e:
                logger.error(f"purge error: {str(e)}")
                self.close()
            return pickle.loads(body) if enable_pickle and body is not None else body
        except Exception as e:
            logger.error(f"recv error: {str(e)}")
            self.close()
        return None

    # -----------------------------
    # send
    # -----------------------------
    def send_once(self, queue: str, body: Any, enable_pickle: bool = True) -> bool:
        try:
            if not self.connect_once():
                return False
            assert self.channel is not None
            body = pickle.dumps(body) if enable_pickle else body
            self.channel.basic_publish(exchange="", routing_key=queue, body=body)
            return True
        except Exception as e:
            logger.error(f"send '{(queue)}' error: {str(e)}")
            self.close()
        return False

    def send_loop(self, queue: str, body: Any, enable_pickle: bool = True) -> bool:
        retry = 0
        while True:
            try:
                self.connect_loop()
                assert self.channel is not None
                body = pickle.dumps(body) if enable_pickle else body
                self.channel.basic_publish(exchange="", routing_key=queue, body=body)
                return True
            except Exception as e:
                retry += 1
                logger.error(f"send '{(queue)}' error: {str(e)}, retry {retry}")
                self.close()
            time.sleep(self.wait_time)

    def send_fanout_once(self, exchange: str, body: Any, enable_pickle: bool = True) -> bool:
        try:
            if not self.connect_once():
                return False
            assert self.channel is not None
            body = pickle.dumps(body) if enable_pickle else body
            self.channel.basic_publish(exchange=exchange, routing_key="", body=body)
            return True
        except Exception as e:
            logger.error(f"send fanout '{(exchange)}' error: {str(e)}")
        return False

    def send_fanout_loop(self, exchange: str, body: Any, enable_pickle: bool = True) -> bool:
        retry = 0
        while True:
            try:
                self.connect_loop()
                assert self.channel is not None
                body = pickle.dumps(body) if enable_pickle else body
                self.channel.basic_publish(exchange=exchange, routing_key="", body=body)
                return True
            except Exception as e:
                retry += 1
                logger.error(f"send fanout '{(exchange)}' error: {str(e)}, retry {retry}")
                self.close()
            time.sleep(self.wait_time)

    # -----------------------------
    # fetch
    # -----------------------------
    def fetch_once(self, queue: str, enable_pickle: bool = True) -> Optional[Any]:
        try:
            if not self.connect_once():
                return None
            assert self.channel is not None
            method_frame, header_frame, body = self.channel.basic_get(queue=queue, auto_ack=False)
            if method_frame is not None:
                self.channel.basic_nack(method_frame.delivery_tag)
                body = pickle.loads(body) if enable_pickle else body
                return body
            return None
        except Exception as e:
            logger.error(f"recv error: {str(e)}")
            self.close()
        return None

    def fetch_messages_once(self, queue: str, enable_pickle: bool = True) -> List[Any]:
        # tagは接続が切れるとリセットされる
        data = []
        try:
            if not self.connect_once():
                return []
            assert self.channel is not None
            tags = []
            while True:
                method_frame, header_frame, body = self.channel.basic_get(queue=queue, auto_ack=False)
                if body is None:
                    break
                data.append(pickle.dumps(body) if enable_pickle else body)
                tags.append(method_frame.delivery_tag)
            [self.channel.basic_nack(tag) for tag in tags]
        except Exception as e:
            logger.error(f"{str(e)}")
            self.close()
        return data

    def fetch_qsize_once(self, queue: str) -> int:
        try:
            if not self.connect_once():
                return -1
            assert self.channel is not None
            queue_info = self.channel.queue_declare(queue=queue, passive=True)
            return queue_info.method.message_count
        except Exception as e:
            logger.error(f"send error: {str(e)}")
        return -1

    def fetch_lastdata_once(self, queue: str, enable_pickle: bool = True) -> Optional[Any]:
        try:
            if not self.connect_once():
                return None
            assert self.channel is not None
            data = None
            tags = []
            while True:
                method_frame, header_frame, body = self.channel.basic_get(queue=queue, auto_ack=False)
                if body is None:
                    break
                data = pickle.loads(body) if enable_pickle else body
                tags.append(method_frame.delivery_tag)
            [self.channel.basic_nack(tag) for tag in tags]
            return data
        except Exception as e:
            logger.error(f"recv error: {str(e)}")
            self.close()
        return None

    # -----------------------------
    # board
    # -----------------------------
    def board_get(self, uid: str):
        self.create_queue_once_if_not_exists(f"board_{uid}")
        return self.fetch_once(f"board_{uid}")

    def board_update(self, board) -> bool:
        # 自分のboardのみupdate可能
        queue = f"board_{self.uid}"
        try:
            if not self.connect_once():
                return False
            assert self.channel is not None
            self.channel.queue_declare(queue=queue)
            self.channel.queue_purge(queue)
            self.channel.basic_publish(exchange="", routing_key=queue, body=pickle.dumps(board))
            logger.debug(f"board update: {str(board)}...")
            return True
        except Exception as e:
            logger.error(f"board_update error: {str(e)}")
            self.close()
        return False

    # -----------------------------
    # manager
    # -----------------------------
    def join(self, role: str):
        self._keepalive_t0 = time.time()

        self.role = role
        self.uid = str(uuid.uuid4())

        # users
        self.create_queue_loop("users")
        self.send_loop("users", json.dumps([self.role, self.uid]), enable_pickle=False)

        # health
        now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.create_queue_loop(f"health_{self.uid}")
        self.send_loop(f"health_{self.uid}", now_str, enable_pickle=False)

        logger.info(f"join: {self.uid}")

    def leave(self):
        try:
            _is_remove_user = True
            for q in self.user_queue_name_list:
                if not self.delete_queue_once(f"{q}_{self.uid}"):
                    _is_remove_user = False

            if _is_remove_user:
                self.remove_users_once([self.uid])

            logger.info(f"leave: {self.uid}")
        except Exception as e:
            logger.error(f"error: {str(e)}")
        self.close()

    def fetch_users(self) -> List[Tuple[str, str]]:
        body_list = self.fetch_messages_once("users", enable_pickle=False)
        return [json.loads(b) for b in body_list]

    def remove_users_once(self, uid_list: List[str]) -> bool:
        # サーバを再接続するとackが切れるので一括実行
        if len(uid_list) == 0:
            return True
        try:
            if not self.connect_once():
                return False
            assert self.channel is not None
            ack_tags = []
            nack_tags = []
            while True:
                method_frame, header_frame, body = self.channel.basic_get(queue="users", auto_ack=False)
                if body is None:
                    break
                role, uid = json.loads(body)
                if uid in uid_list:
                    ack_tags.append(method_frame.delivery_tag)
                else:
                    nack_tags.append(method_frame.delivery_tag)
            [self.channel.basic_ack(tag) for tag in ack_tags]
            [self.channel.basic_nack(tag) for tag in nack_tags]
            return True
        except Exception as e:
            logger.error(f"{str(e)}")
        return False

    def keepalive(self) -> bool:
        if time.time() - self._keepalive_t0 > self.keepalive_interval:
            self._keepalive_t0 = time.time()

            # --- update health
            now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.create_queue_once_if_not_exists(f"health_{self.uid}")
            self.purge_queue_once(f"health_{self.uid}")
            self.send_once(f"health_{self.uid}", now_str, enable_pickle=False)

            # --- usersから消えてたら追加する
            users = self.fetch_users()
            if self.uid not in [uid for role, uid in users]:
                self.create_queue_once_if_not_exists("users")
                self.send_once("users", json.dumps([self.role, self.uid]), enable_pickle=False)

            return True
        return False

    def health_check(self):
        now = datetime.datetime.now()
        logger.debug(f"health check start: {now}")

        # --- other user check
        remove_user_uid_list = []
        users = self.fetch_users()
        for role, uid in users:
            if str(uid) == str(self.uid):
                continue

            # --- 生きているか確認
            _q = f"health_{uid}"
            fail_reason = ""
            if self.exist_queue_once(_q):
                body = self.fetch_once(_q, enable_pickle=False)
                if body is None:
                    fail_reason = "health data recv fail"
                else:
                    # 一定時間以上反応がない場合は死んでると判定
                    date = datetime.datetime.strptime(body.decode(), "%Y%m%d-%H%M%S")
                    diff_seconds = (now - date).total_seconds()
                    if diff_seconds > self.keepalive_limit_time:
                        fail_reason = f"over seconds: {diff_seconds}s > {self.keepalive_limit_time}"
            else:
                fail_reason = "health queue not found"

            # --- 死んでいれば関係あるqを削除する
            if fail_reason != "":
                remove_user_uid_list.append(uid)
                logger.info(f"remove user: {uid}")
                for q in self.user_queue_name_list:
                    self.delete_queue_once(f"{q}_{uid}")

        # --- リストから削除
        self.remove_users_once(remove_user_uid_list)

    def taskcheck_if_my_id_assigned(self):
        client_id = ""
        mp_data = None
        actor_id = 0

        users = self.fetch_users()
        for role, uid in users:
            if role != "client":
                continue
            board = self.board_get(uid)
            if board is None:
                continue
            if board["status"] != "":
                continue

            if self.role == "trainer":
                if board["trainer_id"] == self.uid:
                    client_id = uid
                    mp_data = board["mp_data"]
                    break
            elif self.role == "actor":
                for i, act in enumerate(board["actor_ids"]):
                    if act == self.uid:
                        client_id = uid
                        mp_data = board["mp_data"]
                        actor_id = i
                        break
                if client_id != "":
                    break

        if client_id == "":  # not assign
            return "", None, 0

        # board update
        if self.board_update({"status": "RUN", "client": client_id}):
            return client_id, mp_data, actor_id
        return "", None, 0

    def taskcheck_alive_task(self, client_id):
        board = self.board_get(client_id)
        if board is None:
            return False
        if board["status"] == "END":
            return False
        return True
