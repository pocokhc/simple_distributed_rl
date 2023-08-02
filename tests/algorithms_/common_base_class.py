from abc import ABC, abstractclassmethod
from typing import Union

import pytest

from srl.base.env.config import EnvConfig
from srl.base.rl.config import RLConfig
from srl.runner.runner import Runner
from srl.test.rl import TestRL
from srl.utils import common


class CommonBaseClass(ABC):
    @abstractclassmethod
    def get_framework(self) -> str:
        raise NotImplementedError()

    @abstractclassmethod
    def get_device(self) -> str:
        raise NotImplementedError()

    def check_skip(self):
        if self.get_framework() == "tensorflow":
            pytest.importorskip("tensorflow")

        if self.get_framework() == "torch":
            pytest.importorskip("torch")

    def create_runner(self, env_config: Union[str, EnvConfig], rl_config: RLConfig):
        if isinstance(env_config, str):
            env_config = EnvConfig(env_config)

        device_main = "CPU"
        device_mp_trainer = "CPU"
        device_mp_actors = "CPU"
        if self.get_device() == "GPU":
            if self.get_framework() == "tensorflow":
                if not common.is_available_gpu_tf():
                    pytest.skip()
            if self.get_framework() == "torch":
                if not common.is_available_gpu_tf():
                    pytest.skip()

            device_main = "AUTO"
            device_mp_trainer = "GPU"
            device_mp_actors = "CPU"

        env_config.enable_sanitize_value = False
        rl_config.enable_sanitize_value = False
        runner = Runner(env_config, rl_config)
        runner.set_device(device_main, device_mp_trainer, device_mp_actors)
        runner.set_seed(1)
        runner.set_stats(False)

        return runner, TestRL()
