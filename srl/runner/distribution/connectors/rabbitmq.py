import logging
import pickle
import ssl
import traceback
from typing import Any, Optional, cast

import pika
import pika.exceptions

from srl.runner.distribution.connector_configs import IMemoryReceiver, IMemorySender, IParameterServer, IServerConnector, RabbitMQParameters, RedisParameters

logger = logging.getLogger(__name__)


class _RabbitMQConnector(IServerConnector):
    def __init__(self, parameter: RabbitMQParameters):
        self.parameter = parameter
        self.queue_name = parameter.queue_name

        self.connection = None
        self.channel = None

        self.connection_size = None
        self.channel_size = None

        if self.parameter.url != "":
            self._pika_params = pika.URLParameters(self.parameter.url)
        else:
            self._pika_params = pika.ConnectionParameters(
                self.parameter.host,
                self.parameter.port,
                self.parameter.virtual_host,
                credentials=pika.PlainCredentials(
                    self.parameter.username,
                    self.parameter.password,
                ),
                **self.parameter.kwargs,
            )
        if self.parameter.ssl:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
            ssl_context.set_ciphers("ECDHE+AESGCM:!ECDSA")
            self._pika_params.ssl_options = pika.SSLOptions(context=ssl_context)

    def __del__(self):
        self.close()
        self.close_size

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

    def close_size(self):
        if self.connection_size is not None:
            try:
                self.connection_size.close()
            except pika.exceptions.AMQPError as e:
                logger.error(f"RabbitMQ close error: {str(e)}")

        self.connection_size = None
        self.channel_size = None

    def connect_size(self) -> None:
        if self.connection_size is not None:
            return
        try:
            self.connection_size = pika.BlockingConnection(self._pika_params)
            self.channel_size = self.connection_size.channel()
            return
        except pika.exceptions.AMQPError:
            self.connection_size = None
            self.channel_size = None
            raise

    @property
    def is_connected(self) -> bool:
        return self.connection is not None

    def ping(self) -> bool:
        try:
            if self.channel is None:
                self.connect()
                if self.channel is None:
                    return False
            queue_info = self.channel.queue_declare(queue=self.queue_name, passive=True)
            return queue_info is not None
        except pika.exceptions.AMQPError as e:
            logger.error(e)
            self.close()
            return False
        except Exception as e:
            logger.error(e)
            return False


class RabbitMQReceiver(_RabbitMQConnector, IMemoryReceiver):
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


class RabbitMQSender(_RabbitMQConnector, IMemorySender):
    def memory_send(self, dat: Any) -> None:
        try:
            if self.channel is None:
                self.connect()
                assert self.channel is not None
            dat = pickle.dumps(dat)
            self.channel.basic_publish(exchange="", routing_key=self.queue_name, body=dat)
        except pika.exceptions.AMQPError:
            self.close()
            raise

    def memory_size(self) -> int:
        # なぜか memory_add と memory_size を同じchにすると速度が遅くなる
        try:
            self.connect_size()
            if self.channel_size is None:
                return -1
            queue_info = self.channel_size.queue_declare(queue=self.queue_name, passive=True)
            return queue_info.method.message_count
        except pika.exceptions.ChannelClosed as e:
            logger.info(e)
            self.close_size()
            if e.reply_code == 404:  # NOT_FOUND
                return -1
        except Exception:
            logger.error(traceback.format_exc())
            self.close_size()
        return -1
