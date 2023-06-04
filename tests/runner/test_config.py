import pytest

from srl import runner
from srl.algorithms import dqn
from srl.envs import grid  # noqa F401
from srl.utils import common

common.logger_print()


@pytest.mark.parametrize("framework", ["tensorflow", "torch"])
def test_imported_package(framework):
    # pytest.importorskip は内部でimportしてる可能性あるのでなし
    
    try:
        config = runner.Config("Grid", dqn.Config(framework=framework))
    except ModuleNotFoundError:
        pytest.skip("ModuleNotFoundError")

    if framework == "tensorflow":
        assert config.use_tf
        assert not config.use_torch
    elif framework == "torch":
        assert not config.use_tf
        assert config.use_torch


# pytest.mark.parametrize に記号が入ると動作が変？
@pytest.mark.parametrize("pattern", [i for i in range(12)])
def test_get_device_name_main(pattern):
    run_name, device, distributed, true_device = [
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
    ][pattern]
    config = runner.Config("Grid", None)
    config._run_name = run_name
    config._distributed = distributed
    config.device_main = device
    assert config.get_device_name() == true_device


@pytest.mark.parametrize("pattern", [i for i in range(4)])
def test_get_device_name_trainer(pattern):
    device, true_device = [
        ("", "AUTO"),
        ("AUTO", "AUTO"),
        ("CPU:0", "CPU:0"),
        ("GPU:0", "GPU:0"),
    ][pattern]
    config = runner.Config("Grid", None)
    config._run_name = "trainer"
    config.device_mp_trainer = device
    assert config.get_device_name() == true_device


@pytest.mark.parametrize("pattern", [i for i in range(6)])
def test_get_device_name_actor(pattern):
    device, actor_id, true_device = [
        ("", 0, "CPU"),
        ("AUTO", 0, "CPU"),
        ("GPU:0", 0, "GPU:0"),
        ("CPU:0", 0, "CPU:0"),
        ("CPU:0", 1, "CPU:0"),
        (["CPU:0", "CPU:1"], 1, "CPU:1"),
    ][pattern]
    config = runner.Config("Grid", None)
    config._run_name = f"actor{actor_id}"
    config._actor_id = actor_id
    config.device_mp_actors = device
    assert config.get_device_name() == true_device


def _main_init_device(device, true_device, framework):
    pytest.importorskip(framework)

    config = runner.Config("Grid", dqn.Config(framework=framework))
    config._run_name = "main"
    config._distributed = False
    config.device_main = device

    if framework == "tensorflow":
        import tensorflow  # noqa F401

        config.init_process()
        assert config.used_device_tf == true_device, f"{config.used_device_tf} == {true_device} is fail."

    elif framework == "torch":
        import torch  # noqa F401

        config.init_process()
        assert config.used_device_torch == true_device, f"{config.used_device_torch} == {true_device} is fail."


@pytest.mark.parametrize("pattern", [i for i in range(4)])
def test_main_cpu_init_device(pattern):
    device, true_device, framework = [
        ["cPU", "/CPU", "tensorflow"],
        ["CPU:1", "/CPU:1", "tensorflow"],
        ["CPU", "cpu", "torch"],
        ["CPU:1", "cpu:1", "torch"],
    ][pattern]
    _main_init_device(device, true_device, framework)


@pytest.mark.parametrize("pattern", [i for i in range(2)])
def test_main_tf_gpu_init_device(pattern):
    pytest.importorskip("tensorflow")
    if not common.is_available_gpu_tf():
        pytest.skip()

    device, true_device, framework = [
        ["gPU", "/GPU", "tensorflow"],
        ["GPU:1", "/GPU:1", "tensorflow"],
    ][pattern]
    _main_init_device(device, true_device, framework)


@pytest.mark.parametrize("pattern", [i for i in range(2)])
def test_main_torch_gpu_init_device(pattern):
    pytest.importorskip("torch")
    if not common.is_available_gpu_torch():
        pytest.skip()

    device, true_device = [
        ["gPU", "cuda"],
        ["GPU:1", "cuda:1"],
    ][pattern]
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
