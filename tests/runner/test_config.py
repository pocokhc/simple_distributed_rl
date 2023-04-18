import pytest

from srl import runner
from srl.utils.common import is_available_gpu_tf, is_available_gpu_torch, is_package_installed


@pytest.mark.parametrize(
    "run_name, device, distributed, true_device",
    [
        ("main", "", False, "AUTO"),
        ("main", "", True, "CPU"),
        ("main", "AUTO", False, "AUTO"),
        ("main", "AUTO", True, "CPU"),
        ("main", "CPU:0", False, "CPU:0"),
        ("main", "GPU:0", False, "GPU:0"),
        ("eval", "", False, "AUTO"),
        ("eval", "", True, "CPU"),
        ("eval", "AUTO", False, "AUTO"),
        ("eval", "AUTO", True, "CPU"),
        ("eval", "CPU:0", False, "CPU:0"),
        ("eval", "GPU:0", False, "GPU:0"),
    ],
)
def test_get_device_name_main(run_name, device, distributed, true_device):
    config = runner.Config("Grid", None)
    config.run_name = run_name
    config.device = device
    config.distributed = distributed
    assert config.get_device_name() == true_device


@pytest.mark.parametrize(
    "device, true_device",
    [
        ("", "AUTO"),
        ("AUTO", "AUTO"),
        ("CPU:0", "CPU:0"),
        ("GPU:0", "GPU:0"),
    ],
)
def test_get_device_name_trainer(device, true_device):
    config = runner.Config("Grid", None)
    config.run_name = "trainer"
    config.device_mp_trainer = device
    assert config.get_device_name() == true_device


@pytest.mark.parametrize(
    "device, actor_id, true_device",
    [
        ("", 0, "CPU"),
        ("AUTO", 0, "CPU"),
        ("GPU:0", 0, "GPU:0"),
        ("CPU:0", 0, "CPU:0"),
        ("CPU:0", 1, "CPU:0"),
        (["CPU:0", "CPU:1"], 1, "CPU:1"),
    ],
)
def test_get_device_name_actor(device, actor_id, true_device):
    config = runner.Config("Grid", None)
    config.run_name = f"actor{actor_id}"
    config.run_actor_id = actor_id
    config.device_mp_actor = device
    assert config.get_device_name() == true_device


@pytest.mark.skipif(not is_package_installed("tensorflow"), reason="no module")
def test_init_device_tf_cpu():
    _init_device_assert("CPU", "/CPU", "tf")


@pytest.mark.skipif(not is_package_installed("tensorflow"), reason="no module")
def test_init_device_tf_cpu1():
    _init_device_assert("CPU:1", "/CPU:1", "tf")


@pytest.mark.skipif(not (is_package_installed("tensorflow") and is_available_gpu_tf()), reason="no module")
def test_init_device_tf_auto():
    _init_device_assert("AUTO", "/GPU", "tf")


@pytest.mark.skipif(not (is_package_installed("tensorflow") and is_available_gpu_tf()), reason="no module")
def test_init_device_tf_gpu():
    _init_device_assert("GPU", "/GPU", "tf")


@pytest.mark.skipif(not (is_package_installed("tensorflow") and is_available_gpu_tf()), reason="no module")
def test_init_device_tf_gpu1():
    _init_device_assert("GPU:1", "/GPU:1", "tf")


def _init_device_assert(device, true_device, type):
    config = runner.Config("Grid", None)
    if device == "AUTO":
        config.run_name = "main"
        config.distributed = False
        config.device = "AUTO"
    else:
        config.run_name = "main"
        config.device = device

    from srl.utils import common

    common.logger_print()

    if type == "tf":
        import tensorflow  # noqa F401

        config.init_device()
        assert config.used_device_tf == true_device, f"{config.used_device_tf} == {true_device} is fail."

    elif type == "torch":
        import torch  # noqa F401

        config.init_device()
        assert config.used_device_torch == true_device, f"{config.used_device_torch} == {true_device} is fail."


@pytest.mark.skipif(not is_package_installed("torch"), reason="no module")
def test_init_device_torch_cpu():
    _init_device_assert("CPU", "cpu", "torch")


@pytest.mark.skipif(not is_package_installed("torch"), reason="no module")
def test_init_device_torch_cpu1():
    _init_device_assert("CPU:1", "cpu:1", "torch")


@pytest.mark.skipif(not (is_package_installed("torch") and is_available_gpu_torch()), reason="no module")
def test_init_device_torch_auto():
    _init_device_assert("AUTO", "cuda", "torch")


@pytest.mark.skipif(not (is_package_installed("torch") and is_available_gpu_torch()), reason="no module")
def test_init_device_torch_gpu():
    _init_device_assert("GPU", "cuda", "torch")


@pytest.mark.skipif(not (is_package_installed("torch") and is_available_gpu_torch()), reason="no module")
def test_init_device_torch_gpu1():
    _init_device_assert("GPU:1", "cuda:1", "torch")
