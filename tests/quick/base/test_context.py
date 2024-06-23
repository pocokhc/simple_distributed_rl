import json
from pprint import pprint

import pytest

from srl.algorithms import dqn
from srl.base.context import RunContext, RunNameTypes
from srl.base.env.config import EnvConfig
from srl.utils import common


class NotJsonClass:
    def __init__(self):
        self.a = 1


@pytest.mark.parametrize("framework", ["tensorflow", "torch"])
def test_to_dict(framework):
    if framework == "tensorflow":
        pytest.importorskip("tensorflow")
    elif framework == "torch":
        pytest.importorskip("torch")
    common.logger_print()

    env_config = EnvConfig("Grid")
    rl_config = dqn.Config()
    if framework == "tensorflow":
        rl_config.set_tensorflow()
    elif framework == "torch":
        rl_config.set_torch()

    rl_config.setup(env_config.make())
    c = RunContext()

    c.players = [
        None,
        "AAA",
        ("aa", {"bb": "cc"}),
        dqn.Config(),
        (dqn.Config(), rl_config.make_parameter().backup()),
    ]
    if framework == "tensorflow":
        c.players[3].set_tensorflow()
    elif framework == "torch":
        c.players[3].set_torch()

    def _dummy():
        return 1

    env_config.gym_make_func = _dummy  # type: ignore
    env_config.kwargs = {"a": 1, "not_json_class": NotJsonClass()}

    json_dict = c.to_dict()
    pprint(json_dict)
    json.dumps(json_dict)


# -----------------------------------
# get_device
# -----------------------------------
@pytest.mark.parametrize(
    "run_name, device, true_device",
    [
        [RunNameTypes.main, "AUTo", "AUTO"],
        [RunNameTypes.main, "gpu:1", "GPU:1"],
        [RunNameTypes.trainer, "", "AUTO"],
        [RunNameTypes.eval, "", "CPU"],
    ],
)
def test_get_device(run_name, device, true_device):
    context = RunContext()
    context.run_name = run_name
    context.device = device
    get_device = context.get_device()
    assert get_device == true_device


@pytest.mark.parametrize(
    "device, actor_id, true_device",
    [
        ["AUTo", 0, "CPU"],
        ["GPu", 1, "GPU"],
        [["", ""], 1, "CPU"],
        [["", "AUTO"], 1, "CPU"],
        [["", "GPu"], 1, "GPU"],
        [["CPU:0", "Cpu:1"], 1, "CPU:1"],
    ],
)
def test_get_device_actor(device, actor_id, true_device):
    context = RunContext()
    context.run_name = RunNameTypes.actor
    context.actor_devices = device
    context.actor_id = actor_id
    get_device = context.get_device()
    assert get_device == true_device
