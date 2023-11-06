import logging
import pickle
import ssl
import time
import traceback
from typing import Any, Optional, cast

import pika
import pika.exceptions

from srl.runner.distribution.connectors.imemory import IMemoryConnector
from srl.runner.distribution.connectors.parameters import RabbitMQParameters

logger = logging.getLogger(__name__)


class RabbitMQConnector(IMemoryConnector):
    def __init__(self, parameter: RabbitMQParameters):
        if parameter.url != "":
            self._pika_params = pika.URLParameters(parameter.url)
            if parameter.ssl:
                ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
                ssl_context.set_ciphers("ECDHE+AESGCM:!ECDSA")
                self._pika_params.ssl_options = pika.SSLOptions(context=ssl_context)
        else:
            self._pika_params = pika.ConnectionParameters(
                parameter.host,
                parameter.port,
                parameter.virtual_host,
                credentials=pika.PlainCredentials(
                    parameter.username,
                    parameter.password,
                ),
                **parameter.kwargs,
            )
            if parameter.ssl:
                self._pika_params.ssl_options = pika.SSLOptions(ssl.create_default_context())

        self.connection = None
        self.channel = None
        self.memory_name = ""

    def __del__(self):
        self.close()

    def close(self):
        if self.connection is not None:
            try:
                self.connection.close()
                logger.debug("Connection to RabbitMQ closed")
            except pika.exceptions.AMQPError as e:
                logger.error(f"RabbitMQ close error: {str(e)}")

        self.connection = None
        self.channel = None

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

    def _close_setup(self):
        self.close()
        assert self.memory_name != "", "memory_setup first"
        if self.connect():
            assert self.channel is not None
            self.channel.queue_declare(queue=self.memory_name)
            time.sleep(1)

    def memory_setup(self, task_id: str) -> None:
        self.memory_name = f"task:{task_id}"
        self._close_setup()

    def memory_add(self, dat: Any) -> bool:
        try:
            if self.channel is None:
                self._close_setup()
                assert self.channel is not None
            dat = pickle.dumps(dat)
            self.channel.basic_publish(exchange="", routing_key=self.memory_name, body=dat)
            return True
        except pika.exceptions.AMQPError:
            logger.error(traceback.format_exc())
            self._close_setup()
        return False

    def memory_recv(self) -> Optional[Any]:
        try:
            if self.channel is None:
                self._close_setup()
                assert self.channel is not None
            _, _, body = self.channel.basic_get(queue=self.memory_name, auto_ack=True)
            return body if body is None else pickle.loads(cast(bytes, body))
        except pika.exceptions.AMQPError:
            logger.error(traceback.format_exc())
            self._close_setup()
        return None

    def memory_size(self) -> int:
        try:
            if self.channel is None:
                self._close_setup()
                assert self.channel is not None
            queue_info = self.channel.queue_declare(queue=self.memory_name, passive=True)
            return queue_info.method.message_count
        except pika.exceptions.ChannelClosed as e:
            logger.info(e)
            self._close_setup()
            if e.reply_code == 404:  # NOT_FOUND
                return 0
        except pika.exceptions.AMQPError:
            logger.error(traceback.format_exc())
            self._close_setup()
        return 0

    def memory_delete_if_exist(self, task_id: str) -> None:
        try:
            self.connect()
            assert self.channel is not None
            self.channel.queue_delete(f"task:{task_id}")
        except pika.exceptions.ChannelClosed as e:
            logger.info(e)
            self.close()
        except pika.exceptions.AMQPError:
            self.close()
            raise

    def memory_exist(self, task_id: str) -> bool:
        try:
            self.connect()
            assert self.channel is not None
            self.channel.queue_declare(queue=f"task:{task_id}", passive=True)
            return True
        except pika.exceptions.ChannelClosed as e:
            logger.info(e)
            self.close()
            if e.reply_code == 404:  # NOT_FOUND
                return False
        except pika.exceptions.AMQPError:
            self.close()
            raise
        return False
