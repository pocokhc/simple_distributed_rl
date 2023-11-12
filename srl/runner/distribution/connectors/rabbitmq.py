import logging
import pickle
import ssl
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
        self.queue_name = parameter.queue_name

    def __del__(self):
        self.close()

    def close(self):
        if self.connection is not None:
            try:
                self.connection.close()
            except pika.exceptions.AMQPError as e:
                logger.error(f"RabbitMQ close error: {str(e)}")

        self.connection = None
        self.channel = None

    def connect(self) -> None:
        if self.channel is not None:
            self.channel.queue_declare(queue=self.queue_name)
            return
        try:
            self.connection = pika.BlockingConnection(self._pika_params)
            self.channel = self.connection.channel()
            self.channel.queue_declare(queue=self.queue_name)
            return
        except pika.exceptions.AMQPError:
            self.connection = None
            self.channel = None
            raise

    @property
    def is_connected(self) -> bool:
        return self.connection is not None

    def ping(self) -> bool:
        try:
            self.close()
            self.connect()
        except pika.exceptions.AMQPError as e:
            logger.error(e)
            self.close()
            return False
        except Exception as e:
            logger.error(e)
            return False
        return self.connection is not None

    def memory_add(self, dat: Any) -> None:
        try:
            if self.channel is None:
                self.connect()
                assert self.channel is not None
            dat = pickle.dumps(dat)
            self.channel.basic_publish(exchange="", routing_key=self.queue_name, body=dat)
        except pika.exceptions.AMQPError:
            self.close()
            raise

    def memory_recv(self) -> Optional[Any]:
        try:
            if self.channel is None:
                self.connect()
                assert self.channel is not None
            _, _, body = self.channel.basic_get(queue=self.queue_name, auto_ack=True)
            return body if body is None else pickle.loads(cast(bytes, body))
        except pika.exceptions.AMQPError:
            self.close()
            raise

    def memory_size(self) -> int:
        try:
            if self.channel is None:
                self.connect()
                if self.channel is None:
                    return -1
            queue_info = self.channel.queue_declare(queue=self.queue_name, passive=True)
            return queue_info.method.message_count
        except pika.exceptions.ChannelClosed as e:
            logger.info(e)
            self.close()
            self.connect()
            if e.reply_code == 404:  # NOT_FOUND
                return -1
        except Exception:
            logger.error(traceback.format_exc())
            self.close()
        return -1

    def memory_purge(self) -> None:
        try:
            self.connect()
            if self.channel is None:
                return
            self.channel.queue_purge(self.queue_name)
        except pika.exceptions.ChannelClosed as e:
            logger.info(e)
            self.close()
            self.connect()
            if e.reply_code == 404:  # NOT_FOUND
                return
        except Exception:
            logger.error(traceback.format_exc())
            self.close()
