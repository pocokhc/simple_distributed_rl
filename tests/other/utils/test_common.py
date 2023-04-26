import pytest

from srl.utils import common


def test_is_package_installed():
    assert common.is_package_installed("numpy")
    assert common.is_package_installed("numpy")
    assert not common.is_package_installed("aaaaaa")
    assert not common.is_package_installed("aaaaaa")


def test_is_env_notebook():
    assert not common.is_env_notebook()


def test_compare_less_version():
    assert common.compare_less_version("1.2.a3", "2.0.0")
    assert not common.compare_less_version("2.0.0", "1.2.a3")
    assert not common.compare_less_version("3.0.0", "3.0.0")


def test_is_enable_tf_device_name():
    pytest.importorskip("tensorflow")

    assert common.is_enable_tf_device_name("/CPU:0")
    assert not common.is_enable_tf_device_name("/CPU:99999")
