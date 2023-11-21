import pytest

from srl.runner.distribution.connectors.parameters import MQTTParameters, RabbitMQParameters, RedisParameters
from srl.utils import common
from tests.runner.distribution.memory_test_functions import memory_connector_error_test, memory_connector_test


def test_redis():
    pytest.importorskip("redis")
    common.logger_print()

    memory_connector_test(RedisParameters(host="localhost"))
    memory_connector_error_test(RedisParameters(host="test"))


def test_rabbitmq():
    pytest.importorskip("pika")
    common.logger_print()

    memory_connector_test(RabbitMQParameters(host="localhost", ssl=False))
    memory_connector_error_test(RabbitMQParameters(host="test", ssl=False))


def test_mqtt():
    pytest.importorskip("paho")

    common.logger_print()
    memory_connector_test(MQTTParameters(host="localhost"))
    memory_connector_error_test(MQTTParameters(host="test"))
