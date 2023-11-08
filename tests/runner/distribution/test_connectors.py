import json
import os

import pytest

from srl.runner.distribution.connectors.imemory import IMemoryConnector
from srl.runner.distribution.connectors.parameters import GCPParameters, RabbitMQParameters, RedisParameters
from srl.utils import common


def _load_key():
    key_path = os.path.join(os.path.dirname(__file__), "_env.json")
    keys = json.load(open(key_path))
    return keys


def memory_connector_test(m: IMemoryConnector):
    assert m.ping()

    m.memory_purge()

    # sizeはオプション
    n = m.memory_size()
    if n != -1:
        assert n == 0

    assert m.memory_add({"a": 1})
    n = m.memory_size()
    if n != -1:
        assert m.memory_size() == 1

    d = m.memory_recv()
    assert d is not None
    assert d["a"] == 1
    assert m.memory_recv() is None

    m.memory_purge()

    # sizeはオプション
    n = m.memory_size()
    if n != -1:
        assert n == 0


def test_aiven_redis():
    pytest.importorskip("redis")
    from srl.runner.distribution.connectors.redis_ import RedisConnector

    common.logger_print()
    keys = _load_key()

    m = RedisConnector(RedisParameters(url=keys["aiven_redis_URL"]))
    memory_connector_test(m)


def test_CloudAMQP():
    pytest.importorskip("redis")
    from srl.runner.distribution.connectors.rabbitmq import RabbitMQConnector

    common.logger_print()
    keys = _load_key()

    m = RabbitMQConnector(RabbitMQParameters(url=keys["CloudAMQP_URL"]))
    memory_connector_test(m)


def test_GCP():
    pytest.importorskip("redis")
    from srl.runner.distribution.connectors.gcp import GCPPubSubConnector

    common.logger_print()
    keys = _load_key()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = keys["GCP_CREDENTIALS_FILE_PATH"]

    m = GCPPubSubConnector(GCPParameters(project_id=keys["GCP_project_id"]))
    memory_connector_test(m)


def test_AWS():
    pytest.importorskip("redis")
    from srl.runner.distribution.connectors.rabbitmq import RabbitMQConnector

    common.logger_print()
    keys = _load_key()

    m = RabbitMQConnector(RabbitMQParameters(url=keys["AWS_URL"]))
    memory_connector_test(m)
