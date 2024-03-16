import pytest

from srl.utils import common
try:
    from tests.runner.distribution.memory_test_functions import memory_connector_error_test, memory_connector_test
except ModuleNotFoundError as e:
    print(e)


def test_redis():
    pytest.importorskip("redis")
    common.logger_print()
    from srl.runner.distribution.connectors.parameters import RedisParameters

    memory_connector_test(RedisParameters(host="localhost"))
    memory_connector_error_test(RedisParameters(host="test"))


def test_rabbitmq():
    pytest.importorskip("pika")
    common.logger_print()
    from srl.runner.distribution.connectors.parameters import RabbitMQParameters

    memory_connector_test(RabbitMQParameters(host="localhost", ssl=False))
    memory_connector_error_test(RabbitMQParameters(host="test", ssl=False))


def test_mqtt():
    pytest.importorskip("paho")
    common.logger_print()
    from srl.runner.distribution.connectors.parameters import MQTTParameters

    memory_connector_test(MQTTParameters(host="localhost"))
    memory_connector_error_test(MQTTParameters(host="test"))
