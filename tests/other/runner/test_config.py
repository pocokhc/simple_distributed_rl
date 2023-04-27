import pytest

from srl import runner
from srl.utils import common

common.logger_print()

# pytest.mark.parametrize に記号が入ると動作が変？


def test_get_device_name_main():
    for run_name, device, distributed, true_device in [
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
    ]:
        config = runner.Config("Grid", None)
        config.run_name = run_name
        config.device_main = device
        config.distributed = distributed
        assert config.get_device_name() == true_device


def test_get_device_name_trainer():
    for device, true_device in [
        ("", "AUTO"),
        ("AUTO", "AUTO"),
        ("CPU:0", "CPU:0"),
        ("GPU:0", "GPU:0"),
    ]:
        config = runner.Config("Grid", None)
        config.run_name = "trainer"
        config.device_mp_trainer = device
        assert config.get_device_name() == true_device


def test_get_device_name_actor():
    for device, actor_id, true_device in [
        ("", 0, "CPU"),
        ("AUTO", 0, "CPU"),
        ("GPU:0", 0, "GPU:0"),
        ("CPU:0", 0, "CPU:0"),
        ("CPU:0", 1, "CPU:0"),
        (["CPU:0", "CPU:1"], 1, "CPU:1"),

    ]:
        config = runner.Config("Grid", None)
        config.run_name = f"actor{actor_id}"
        config.run_actor_id = actor_id
        config.device_mp_actors = device
        assert config.get_device_name() == true_device


def _main_init_device(device, true_device, module):
    pytest.importorskip(module)

    config = runner.Config("Grid", None)
    config.run_name = "main"
    config.distributed = False
    if device == "AUTO":
        config.device_main = "AUTO"
    else:
        config.device_main = device

    if module == "tensorflow":
        import tensorflow  # noqa F401

        config.init_device()
        assert config.used_device_tf == true_device, f"{config.used_device_tf} == {true_device} is fail."

    elif module == "torch":
        import torch  # noqa F401

        config.init_device()
        assert config.used_device_torch == true_device, f"{config.used_device_torch} == {true_device} is fail."


def test_main_cpu_init_device():
    for device, true_device, module in [
        ["cPU", "/CPU", "tensorflow"],
        ["CPU:1", "/CPU:1", "tensorflow"],
        ["CPU", "cpu", "torch"],
        ["CPU:1", "cpu:1", "torch"],
    ]:
        _main_init_device(device, true_device, module)


def test_main_tf_gpu_init_device():
    pytest.importorskip("tensorflow")
    if not common.is_available_gpu_tf():
        pytest.skip()

    for device, true_device, module in [
        ["gPU", "/GPU", "tensorflow"],
        ["GPU:1", "/GPU:1", "tensorflow"],
    ]:
        _main_init_device(device, true_device, module)


def test_main_torch_gpu_init_device():
    pytest.importorskip("torch")
    if not common.is_available_gpu_torch():
        pytest.skip()
    for device, true_device in [
        ["gPU", "cuda"],
        ["GPU:1", "cuda:1"],
    ]:
        _main_init_device(device, true_device, "torch")


def test_main_tf_auto_cpu_init_device():
    pytest.importorskip("tensorflow")
    if common.is_available_gpu_tf():
        pytest.skip()
    _main_init_device("AUTO", "/CPU", "tensorflow")


def test_main_torch_auto_cpu_init_device():
    pytest.importorskip("torch")
    if common.is_available_gpu_torch():
        pytest.skip()
    _main_init_device("AUTO", "cpu", "torch")


def test_main_tf_auto_gpu_init_device():
    pytest.importorskip("tensorflow")
    if not common.is_available_gpu_tf():
        pytest.skip()
    _main_init_device("AUTO", "/GPU", "tensorflow")


def test_main_torch_auto_gpu_init_device():
    pytest.importorskip("torch")
    if not common.is_available_gpu_torch():
        pytest.skip()
    _main_init_device("AUTO", "cuda", "torch")
