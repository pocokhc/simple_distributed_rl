import pytest

from srl.runner.distribution.connector_configs import MQTTParameters, RabbitMQParameters, RedisParameters
from tests.distribution.runner.distribution_external.memory_test_functions import (
    memory_connector_error_test,
    memory_connector_test,
)


def test_redis():
    pytest.importorskip("redis")
    memory_connector_test(RedisParameters(host="localhost"))


def test_redis_error():
    pytest.importorskip("redis")
    memory_connector_error_test(RedisParameters(host="test"))


def test_rabbitmq():
    pytest.importorskip("pika")
    memory_connector_test(RabbitMQParameters(host="localhost", ssl=False))


def test_rabbitmq_error():
    pytest.importorskip("pika")
    memory_connector_error_test(RabbitMQParameters(host="test", ssl=False))


def test_mqtt():
    pytest.importorskip("paho")
    memory_connector_test(MQTTParameters(host="localhost"))


def test_mqtt_error():
    pytest.importorskip("paho")
    memory_connector_error_test(MQTTParameters(host="test"))
