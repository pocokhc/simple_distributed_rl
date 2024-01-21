from abc import ABC, abstractmethod
from typing import Tuple, Union

import pytest

import srl
from srl.base.env.config import EnvConfig
from srl.base.rl.config import RLConfig
from srl.envs import grid
from srl.test.rl import TestRL
from srl.utils import common


class CommonBaseSimpleTest:
    @pytest.fixture()
    def rl_param(self, request):
        return None

    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        raise NotImplementedError()

    def test_simple_check(self, rl_param):
        rl_config, simple_check_kwargs = self.create_rl_config(rl_param)
        tester = TestRL()
        tester.simple_check(rl_config, **simple_check_kwargs)

    def test_simple_check_mp(self, rl_param):
        rl_config, simple_check_kwargs = self.create_rl_config(rl_param)
        tester = TestRL()
        tester.simple_check(rl_config, is_mp=True, **simple_check_kwargs)

    def test_summary(self, rl_param):
        rl_config, simple_check_kwargs = self.create_rl_config(rl_param)
        _rl_config = rl_config.copy(reset_env_config=True)

        if simple_check_kwargs.get("use_layer_processor", False):
            _rl_config.processors.append(grid.LayerProcessor())

        _rl_config.setup(srl.make_env("Grid"))
        parameter = srl.make_parameter(_rl_config)
        parameter.summary()


class CommonBaseClass(ABC):
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

        env_config.enable_sanitize_value = False
        rl_config.enable_sanitize_value = False
        runner = srl.Runner(env_config, rl_config)
        runner.set_device(device)
        runner.set_seed(1)
        runner.disable_stats()
        runner.set_progress_options(interval_limit=30)

        return runner, TestRL()
