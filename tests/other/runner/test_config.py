import pytest

from srl import runner
from srl.utils import common

common.logger_print()


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
    config.device_main = device
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


@pytest.mark.parametrize(
    "device, true_device, module",
    [
        ["cPU", "/CPU", "tensorflow"],
        ["CPU:1", "/CPU:1", "tensorflow"],
        ["CPU", "cpu", "torch"],
        ["CPU:1", "cpu:1", "torch"],
    ],
)
def test_main_cpu_init_device(device, true_device, module):
    _main_init_device(device, true_device, module)


@pytest.mark.parametrize(
    "device, true_device",
    [
        ["gPU", "/GPU"],
        ["GPU:1", "/GPU:1"],
    ],
)
def test_main_tf_gpu_init_device(device, true_device):
    pytest.importorskip("tensorflow")
    if not common.is_available_gpu_tf():
        pytest.skip()
    _main_init_device(device, true_device, "tensorflow")


@pytest.mark.parametrize(
    "device, true_device, module",
    [
        ["gPU", "cuda", "torch"],
        ["GPU:1", "cuda:1", "torch"],
    ],
)
def test_main_torch_gpu_init_device(device, true_device, module):
    pytest.importorskip("torch")
    if not common.is_available_gpu_torch():
        pytest.skip()
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
