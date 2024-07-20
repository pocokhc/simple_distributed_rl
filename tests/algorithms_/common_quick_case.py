from abc import ABC, abstractmethod
from typing import Tuple, cast

import pytest

import srl
from srl.base.define import ObservationModes
from srl.base.rl.config import RLConfig
from srl.rl.models.config.framework_config import RLConfigComponentFramework
from srl.test.rl import TestRL
from srl.utils import common


class CommonQuickCase(ABC):
    @pytest.fixture()
    def rl_param(self, request):
        return None

    @abstractmethod
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        raise NotImplementedError()

    @abstractmethod
    def use_framework(self) -> str:
        raise NotImplementedError()

    def check_skip(self):
        if self.use_framework() == "tensorflow":
            pytest.importorskip("tensorflow")

        if self.use_framework() == "torch":
            pytest.importorskip("torch")

    def _setup_rl_config(self, rl_config: RLConfig):
        common.logger_print()

        rl_config.memory_compress = False
        if issubclass(rl_config.__class__, RLConfigComponentFramework):
            if self.use_framework() == "tensorflow":
                cast(RLConfigComponentFramework, rl_config).set_tensorflow()
            elif self.use_framework() == "torch":
                cast(RLConfigComponentFramework, rl_config).set_torch()

    def test_simple(self, rl_param):
        common.logger_print()
        self.check_skip()
        rl_config, test_kwargs = self.create_rl_config(rl_param)
        self._setup_rl_config(rl_config)

        tester = TestRL()
        tester.test(rl_config, **test_kwargs)

    def test_simple_mp(self, rl_param):
        common.logger_print()
        self.check_skip()
        rl_config, test_kwargs = self.create_rl_config(rl_param)
        self._setup_rl_config(rl_config)

        tester = TestRL()
        tester.test(rl_config, test_mp=True, **test_kwargs)

    def test_input_image(self, rl_param):
        pytest.importorskip("PIL")
        pytest.importorskip("pygame")
        common.logger_print()
        self.check_skip()
        rl_config, test_kwargs = self.create_rl_config(rl_param)
        self._setup_rl_config(rl_config)

        rl_config.observation_mode = ObservationModes.RENDER_IMAGE
        tester = TestRL()
        tester.test(rl_config, **test_kwargs)

    def test_input_multi(self, rl_param):
        self.check_skip()
        pytest.skip("TODO")
        common.logger_print()
        rl_config, test_kwargs = self.create_rl_config(rl_param)
        self._setup_rl_config(rl_config)

        rl_config.observation_mode = ObservationModes.ENV | ObservationModes.RENDER_IMAGE
        tester = TestRL()
        tester.test(rl_config, **test_kwargs)

    def test_summary(self, rl_param):
        common.logger_print()
        self.check_skip()
        rl_config, test_kwargs = self.create_rl_config(rl_param)
        self._setup_rl_config(rl_config)

        env_config = srl.EnvConfig("Grid")
        if test_kwargs.get("use_layer_processor", False):
            env_config.kwargs["obs_type"] = "layer"
        env = env_config.make()

        parameter = rl_config.make_parameter(env)
        parameter.summary()
