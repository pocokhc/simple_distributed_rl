from abc import ABC, abstractclassmethod

import pytest

from srl import runner
from srl.test.rl import TestRL
from srl.utils import common


class CommonBaseClass(ABC):
    @abstractclassmethod
    def get_framework(self) -> str:
        raise NotImplementedError()

    @abstractclassmethod
    def get_device(self) -> str:
        raise NotImplementedError()

    def create_config(self, env_config, rl_config):
        if self.get_framework() == "tensorflow":
            pytest.importorskip("tensorflow")

        if self.get_framework() == "torch":
            pytest.importorskip("torch")

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
        config = runner.Config(
            env_config,
            rl_config,
            device_main=device_main,
            device_mp_trainer=device_mp_trainer,
            device_mp_actors=device_mp_actors,
            seed=1,
            seed_enable_gpu=True,
        )
        config.env_config.enable_sanitize_value = False
        return config, TestRL()
