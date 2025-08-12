import logging
import pickle
import queue
from typing import Any, Optional

import paho.mqtt.client as mqtt

from srl.runner.distribution.connector_configs import IMemoryReceiver, IMemorySender, IParameterServer, IServerConnector, MQTTParameters, RedisParameters

logger = logging.getLogger(__name__)


class _MQTTConnector(IServerConnector):
    def __init__(self, parameter: MQTTParameters):
        self.parameter = parameter
        self.client: Optional[mqtt.Client] = None

    def __del__(self):
        self.close()

    def close(self):
        if self.client is not None:
            try:
                self.client.disconnect()
            except Exception as e:
                logger.error(f"MQTT disconnect error: {str(e)}")
        self.client = None

    def connect(self) -> None:
        try:
            self.client = mqtt.Client()
            self.client.connect(host=self.parameter.host, port=self.parameter.port, **self.parameter.kwargs)
            return
        except Exception:
            self.client = None
            raise

    @property
    def is_connected(self) -> bool:
        return self.client is not None

    def ping(self) -> bool:
        try:
            self.close()
            self.connect()
        except Exception as e:
            logger.error(e)
            return False
        return self.client is not None


class MQTTReceiver(_MQTTConnector, IMemoryReceiver):
    def __init__(self, parameter: MQTTParameters):
        super().__init__(parameter)

        self.queue = queue.Queue()

    def connect(self) -> None:
        try:
            super().connect()
            assert self.client is not None
            self.client.subscribe(self.parameter.topic_name)
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            self.client.loop_start()
            return
        except Exception:
            self.client = None
            raise

    def _on_disconnect(self, client, userdata, rc):
        self.client = None

    def _on_message(self, client, userdata, msg):
        self.queue.put(pickle.loads(msg.payload))

    def memory_recv(self) -> Optional[Any]:
        assert self.client is not None
        if self.queue.empty():
            return None
        return self.queue.get(timeout=1)

    def memory_purge(self) -> None:
        self.queue = queue.Queue()


class MQTTSender(_MQTTConnector, IMemorySender):
    def memory_send(self, dat: Any) -> None:
        try:
            if self.client is None:
                self.connect()
                assert self.client is not None
            dat = pickle.dumps(dat)
            self.client.publish(self.parameter.topic_name, dat)
        except Exception:
            self.close()
            raise

    def memory_size(self) -> int:
        return -1
