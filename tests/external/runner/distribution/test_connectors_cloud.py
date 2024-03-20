import json
import os

import pytest

from tests.external.runner.distribution.memory_test_functions import memory_connector_test

try:
    from srl.runner.distribution.connectors.parameters import GCPParameters, RabbitMQParameters, RedisParameters
    from srl.utils import common
except ModuleNotFoundError as e:
    print(e)


def _load_key():
    key_path = os.path.join(os.path.dirname(__file__), "_env.json")
    keys = json.load(open(key_path))
    return keys


def test_aiven_redis():
    pytest.importorskip("redis")

    common.logger_print()
    keys = _load_key()

    m = RedisParameters(url=keys["aiven_redis_URL"])
    memory_connector_test(m)


def test_CloudAMQP():
    pytest.importorskip("pika")

    common.logger_print()
    keys = _load_key()

    m = RabbitMQParameters(url=keys["CloudAMQP_URL"])
    memory_connector_test(m)


def test_GCP():
    pytest.importorskip("google")

    common.logger_print()
    keys = _load_key()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = keys["GCP_CREDENTIALS_FILE_PATH"]

    m = GCPParameters(project_id=keys["GCP_project_id"])
    memory_connector_test(m)


def test_AWS():
    pytest.importorskip("redis")

    common.logger_print()
    keys = _load_key()

    m = RabbitMQParameters(url=keys["AWS_URL"])
    memory_connector_test(m)
