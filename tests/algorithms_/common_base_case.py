from abc import ABC, abstractmethod
from typing import Union

import pytest

import srl
from srl.base.env.config import EnvConfig
from srl.base.rl.config import RLConfig
from srl.test.rl import TestRL
from srl.utils import common


class CommonBaseCase(ABC):
    @abstractmethod
    def get_framework(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def get_device(self) -> str:
        raise NotImplementedError()

    def check_skip(self):
        if self.get_framework() == "tensorflow":
            pytest.importorskip("tensorflow")

        if self.get_framework() == "torch":
            pytest.importorskip("torch")

    def create_runner(self, env_config: Union[str, EnvConfig], rl_config: RLConfig):
        common.logger_print()
        if isinstance(env_config, str):
            env_config = EnvConfig(env_config)

        device = "CPU"
        if self.get_device() == "GPU":
            if self.get_framework() == "tensorflow":
                if not common.is_available_gpu_tf():
                    pytest.skip()
            if self.get_framework() == "torch":
                if not common.is_available_gpu_torch():
                    pytest.skip()

            device = "GPU"

        if self.get_framework() == "tensorflow":
            rl_config.set_tensorflow()
        elif self.get_framework() == "torch":
            rl_config.set_torch()

        rl_config.memory_compress = False
        env_config.enable_sanitize = False
        rl_config.enable_sanitize = False
        runner = srl.Runner(env_config, rl_config)
        runner.set_device(device)
        runner.set_seed(1)
        runner.disable_stats()
        runner.set_progress_options(interval_limit=30)

        return runner, TestRL()
