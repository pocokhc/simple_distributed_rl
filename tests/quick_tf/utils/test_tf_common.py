import pytest

from srl.utils import common


def test_is_enable_tf_device_name():
    pytest.importorskip("tensorflow")

    assert common.is_enable_tf_device_name("/CPU:0")
    assert not common.is_enable_tf_device_name("/CPU:99999")
