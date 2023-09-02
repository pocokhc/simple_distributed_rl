import os

import pytest

from srl.algorithms import dqn, ql
from srl.runner.runner import Runner
from srl.utils import common

common.logger_print()


def _main_init_device(
    framework,
    run_name,
    distributed,
    device_main="AUTO",
    device_mp_actors="AUTO",
    device_mp_trainer="AUTO",
    actor_id=0,
    true_tf="/CPU",
    true_torch="cpu",
    is_assert=False,
):
    if framework == "tensorflow":
        pytest.importorskip("tensorflow")
        rl_config = dqn.Config()
        rl_config.framework.set_tensorflow()
        runner = Runner("Grid", rl_config)
    elif framework == "torch":
        pytest.importorskip("torch")
        rl_config = dqn.Config()
        rl_config.framework.set_torch()
        runner = Runner("Grid", rl_config)
    else:
        runner = Runner("Grid", ql.Config())

    Runner._Runner__is_init_device = False  # type: ignore
    runner.set_device(
        device_main=device_main,
        device_mp_actors=device_mp_actors,
        device_mp_trainer=device_mp_trainer,
    )
    runner.context.run_name = run_name
    runner.context.distributed = distributed
    runner.context.actor_id = actor_id
    # --------------

    if is_assert:
        with pytest.raises(AssertionError):
            runner._init_process()
        return
    else:
        runner._init_process()

    assert runner.context.framework == framework
    if framework == "tensorflow" or framework == "":
        assert runner.context.used_device_tf == true_tf
        assert runner.rl_config.used_device_tf == true_tf
    if framework == "torch" or framework == "":
        assert runner.context.used_device_torch == true_torch
        assert runner.rl_config.used_device_torch == true_torch


@pytest.mark.parametrize("run_name", ["main", "eval", "trainer", "actor0"])
@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("device", ["", "AUTO", "CPU", "GPU", "CPU:0", "GPU:0"])
def test_init_device_no_framework(run_name, distributed, device):
    _main_init_device(
        framework="",
        run_name=run_name,
        distributed=distributed,
        device_main=device,
        true_tf="/CPU",
        true_torch="cpu",
    )


@pytest.mark.parametrize("framework", ["tensorflow", "torch"])
@pytest.mark.parametrize("run_name", ["main", "eval"])
@pytest.mark.parametrize(
    "device, device_main, distributed, true_tf, true_torch, is_assert",
    [
        ["CPU", "AUTO", False, "/CPU", "cpu", False],
        ["GPU", "AUTO", False, "/GPU", "cuda", False],
        ["CPU", "AUTO", True, "/CPU", "cpu", False],
        ["GPU", "AUTO", True, "/CPU", "cpu", False],
        ["CPU", "CpU", False, "/CPU", "cpu", False],
        ["GPU", "CpU", False, "/CPU", "cpu", False],
        ["CPU", "Gpu", False, "/CPU", "cpu", True],
        ["GPU", "Gpu", False, "/GPU", "cuda", False],
        ["CPU", "cpu:1", False, "/CPU:1", "cpu:1", False],
        ["GPU", "cpu:1", False, "/CPU:1", "cpu:1", False],
        ["CPU", "Gpu:2", False, "/CPU:2", "cpu", True],
        ["GPU", "Gpu:2", False, "/GPU:2", "cuda:2", False],
    ],
)
def test_init_device(
    device,
    framework,
    run_name,
    distributed,
    device_main,
    true_tf,
    true_torch,
    is_assert,
):
    if device == "CPU":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-2"  # change CPU
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "-2":
            del os.environ["CUDA_VISIBLE_DEVICES"]
        if framework == "tensorflow":
            if not common.is_available_gpu_tf():
                pytest.skip()
        elif framework == "torch":
            if not common.is_available_gpu_torch():
                pytest.skip()

    _main_init_device(
        framework=framework,
        run_name=run_name,
        distributed=distributed,
        device_main=device_main,
        true_tf=true_tf,
        true_torch=true_torch,
        is_assert=is_assert,
    )


@pytest.mark.parametrize("framework", ["tensorflow", "torch"])
@pytest.mark.parametrize(
    "device, device_mp_trainer, true_tf, true_torch, is_assert",
    [
        ["CPU", "AUTO", "/CPU", "cpu", False],
        ["GPU", "AUTO", "/GPU", "cuda", False],
        ["CPU", "CpU", "/CPU", "cpu", False],
        ["GPU", "CpU", "/CPU", "cpu", False],
        ["CPU", "Gpu", "/CPU", "cpu", True],
        ["GPU", "Gpu", "/GPU", "cuda", False],
        ["CPU", "cpu:1", "/CPU:1", "cpu:1", False],
        ["GPU", "cpu:1", "/CPU:1", "cpu:1", False],
        ["CPU", "Gpu:2", "/CPU", "cpu", True],
        ["GPU", "Gpu:2", "/GPU:2", "cuda:2", False],
    ],
)
def test_init_device_trainer(device, framework, device_mp_trainer, true_tf, true_torch, is_assert):
    if device == "CPU":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-2"  # change CPU
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "-2":
            del os.environ["CUDA_VISIBLE_DEVICES"]
        if framework == "tensorflow":
            if not common.is_available_gpu_tf():
                pytest.skip()
        elif framework == "torch":
            if not common.is_available_gpu_torch():
                pytest.skip()

    _main_init_device(
        framework=framework,
        run_name="trainer",
        distributed=True,
        device_mp_trainer=device_mp_trainer,
        true_tf=true_tf,
        true_torch=true_torch,
        is_assert=is_assert,
    )


@pytest.mark.parametrize("framework", ["tensorflow", "torch"])
@pytest.mark.parametrize(
    "device, device_mp_actors, actor_id, true_tf, true_torch, is_assert",
    [
        ["CPU", "AUTO", 1, "/CPU", "cpu", False],
        ["GPU", "AUTO", 1, "/CPU", "cpu", False],
        ["CPU", "CpU", 1, "/CPU", "cpu", False],
        ["GPU", "CpU", 1, "/CPU", "cpu", False],
        ["CPU", "Gpu", 1, "/CPU", "cpu", True],
        ["GPU", "Gpu", 1, "/GPU", "cuda", False],
        ["CPU", "cpu:1", 1, "/CPU:1", "cpu:1", False],
        ["GPU", "cpu:1", 1, "/CPU:1", "cpu:1", False],
        ["CPU", "Gpu:2", 1, "/CPU", "cpu", True],
        ["GPU", "Gpu:2", 1, "/GPU:2", "cuda:2", False],
        ["CPU", ["CPU:0", "Cpu:1"], 1, "/CPU:1", "cpu:1", False],
    ],
)
def test_init_device_actor(device, framework, device_mp_actors, actor_id, true_tf, true_torch, is_assert):
    if device == "CPU":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-2"  # change CPU
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "-2":
            del os.environ["CUDA_VISIBLE_DEVICES"]
        if framework == "tensorflow":
            if not common.is_available_gpu_tf():
                pytest.skip()
        elif framework == "torch":
            if not common.is_available_gpu_torch():
                pytest.skip()

    _main_init_device(
        framework=framework,
        run_name="actor" + str(actor_id),
        distributed=True,
        device_mp_actors=device_mp_actors,
        true_tf=true_tf,
        true_torch=true_torch,
        actor_id=actor_id,
        is_assert=is_assert,
    )


@pytest.mark.parametrize(
    "use_CUDA_VISIBLE_DEVICES, device, true_environ",
    [
        [True, "CPU", "-1"],
        [True, "GPU", None],  # no GPU error OK
        [False, "CPU", None],
        [False, "GPU", None],  # no GPU error OK
    ],
)
def test_use_CUDA_VISIBLE_DEVICES(use_CUDA_VISIBLE_DEVICES, device, true_environ):
    rl_config = dqn.Config()

    try:
        if rl_config.get_use_framework() == "tensorflow":
            pytest.importorskip("tensorflow")
        elif rl_config.get_use_framework() == "torch":
            pytest.importorskip("torch")
    except AssertionError:
        pytest.skip()

    # 一括で実行する場合はNGになる
    runner = Runner("Grid", rl_config)

    Runner._Runner__is_init_device = False  # type: ignore
    runner.set_device(device, use_CUDA_VISIBLE_DEVICES=use_CUDA_VISIBLE_DEVICES)
    runner._init_process()

    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    print(cuda)

    assert cuda == true_environ


def test_play2_tf():
    if not common.is_available_gpu_tf():
        pytest.skip()

    rl_config = dqn.Config()
    rl_config.framework.set_tensorflow()

    runner = Runner("Grid", rl_config)
    runner.train(max_episodes=1)
    assert runner.context.used_device_tf == "/GPU"

    runner = Runner("Grid", rl_config)
    runner.train(max_episodes=1)
    assert runner.context.used_device_tf == "/GPU"


def test_play2_torch():
    if not common.is_available_gpu_torch():
        pytest.skip()

    rl_config = dqn.Config()
    rl_config.framework.set_torch()

    runner = Runner("Grid", rl_config)
    runner.train(max_episodes=1)
    assert runner.context.used_device_torch == "cuda"

    runner = Runner("Grid", rl_config)
    runner.train(max_episodes=1)
    assert runner.context.used_device_torch == "cuda"
