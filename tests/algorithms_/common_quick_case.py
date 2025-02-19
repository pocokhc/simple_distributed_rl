from abc import ABC, abstractmethod
from typing import Tuple, cast

import pytest

import srl
from srl.base.define import ObservationModes
from srl.base.rl.config import RLConfig
from srl.rl.models.config.framework_config import RLConfigComponentFramework
from srl.test import rl as test_rl
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

    def use_device(self) -> str:
        return "AUTO"

    def _check_test_params(self):
        if self.use_framework() == "tensorflow":
            pytest.importorskip("tensorflow")
        elif self.use_framework() == "torch":
            pytest.importorskip("torch")
        elif self.use_framework() == "":
            pass
        else:
            raise ValueError(self.use_framework())

        if self.use_device() == "CPU":
            pass
        elif self.use_device() == "GPU":
            if self.use_framework() == "tensorflow":
                assert common.is_available_gpu_tf()
            elif self.use_framework() == "torch":
                assert common.is_available_gpu_torch()
        elif self.use_device() == "AUTO":
            pass
        else:
            raise ValueError(self.use_device())

    def _setup_rl_config(self, rl_config: RLConfig):
        rl_config.memory_compress = False
        if issubclass(rl_config.__class__, RLConfigComponentFramework):
            if self.use_framework() == "tensorflow":
                cast(RLConfigComponentFramework, rl_config).set_tensorflow()
            elif self.use_framework() == "torch":
                cast(RLConfigComponentFramework, rl_config).set_torch()

    def test_simple(self, rl_param, tmpdir):
        self._check_test_params()
        rl_config, test_kwargs = self.create_rl_config(rl_param)
        self._setup_rl_config(rl_config)
        test_rl.test_rl(rl_config, device=self.use_device(), **test_kwargs, tmp_dir=tmpdir)

    def test_simple_mp(self, rl_param, tmpdir):
        self._check_test_params()
        rl_config, test_kwargs = self.create_rl_config(rl_param)
        self._setup_rl_config(rl_config)
        test_rl.test_rl(
            rl_config,
            device=self.use_device(),
            test_mp=True,
            test_render_terminal=False,
            test_render_window=False,
            **test_kwargs,
            tmp_dir=tmpdir,
        )

    def test_simple_input_image(self, rl_param, tmpdir):
        pytest.importorskip("PIL")
        pytest.importorskip("pygame")
        self._check_test_params()
        rl_config, test_kwargs = self.create_rl_config(rl_param)
        self._setup_rl_config(rl_config)
        rl_config.observation_mode = ObservationModes.RENDER_IMAGE
        test_rl.test_rl(rl_config, device=self.use_device(), **test_kwargs, tmp_dir=tmpdir)

    def test_summary(self, rl_param):
        self._check_test_params()
        rl_config, test_kwargs = self.create_rl_config(rl_param)
        self._setup_rl_config(rl_config)

        if test_kwargs.get("use_layer_processor", False):
            env_config = srl.EnvConfig("Grid-layer")
        else:
            env_config = srl.EnvConfig("Grid")
        env = env_config.make()

        parameter = rl_config.make_parameter(env)
        parameter.summary()
