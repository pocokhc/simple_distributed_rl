from abc import ABC, abstractmethod
from typing import Union, cast

import pytest

import srl
from srl.base.env.config import EnvConfig
from srl.base.rl.config import RLConfig
from srl.rl.models.config.framework_config import RLConfigComponentFramework


class CommonLongCase(ABC):
    @abstractmethod
    def use_framework(self) -> str:
        raise NotImplementedError()

    def use_device(self) -> str:
        return "AUTO"

    def check_test_skip(self):
        if self.use_framework() == "tensorflow":
            pytest.importorskip("tensorflow")
        elif self.use_framework() == "torch":
            pytest.importorskip("torch")
        elif self.use_framework() == "":
            pass
        else:
            raise ValueError(self.use_framework())

    def create_test_runner(self, env_config: Union[str, EnvConfig], rl_config: RLConfig, seed: int = 1):
        if isinstance(env_config, str):
            env_config = EnvConfig(env_config)

        if issubclass(rl_config.__class__, RLConfigComponentFramework):
            if self.use_framework() == "tensorflow":
                cast(RLConfigComponentFramework, rl_config).set_tensorflow()
            elif self.use_framework() == "torch":
                cast(RLConfigComponentFramework, rl_config).set_torch()

        rl_config.memory_compress = False
        env_config.enable_sanitize = False
        rl_config.enable_sanitize = False
        runner = srl.Runner(env_config, rl_config)
        runner.set_device(self.use_device())
        runner.set_seed(seed)
        runner.disable_stats()
        runner.set_progress(interval_limit=30)
        return runner
