import unittest
from typing import Tuple

import pytest

from srl.base.rl.config import RLConfig
from srl.test import TestRL
from tests.algorithms_.common_base_class import CommonBaseSimpleTest


class Test_agent57(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("torch")

        from srl.algorithms import agent57

        rl_config = agent57.Config()
        rl_config.framework.set_torch()
        rl_config.memory_warmup_size = 100
        return rl_config, {}


class Test_agent57_light(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("torch")

        from srl.algorithms import agent57_light

        rl_config = agent57_light.Config()
        rl_config.framework.set_torch()
        rl_config.memory_warmup_size = 100
        return rl_config, {}


class Test_dqn(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("torch")

        from srl.algorithms import dqn

        rl_config = dqn.Config()
        rl_config.framework.set_torch()
        return rl_config, {}

    def test_simple_check_atari_config(self):
        pytest.importorskip("torch")

        from srl.algorithms import dqn

        rl_config = dqn.Config()
        rl_config.framework.set_torch()
        rl_config.use_render_image_for_observation = True
        rl_config.set_atari_config()
        rl_config.memory_warmup_size = 1000
        tester = TestRL()
        tester.simple_check(rl_config)


class Test_rainbow(CommonBaseSimpleTest):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        pytest.importorskip("torch")

        from srl.algorithms import rainbow

        rl_config = rainbow.Config()
        rl_config.framework.set_torch()
        return rl_config, {}

    def test_simple_check_atari_config(self):
        pytest.importorskip("torch")

        from srl.algorithms import rainbow

        rl_config = rainbow.Config()
        rl_config.framework.set_torch()
        rl_config.set_atari_config()
        rl_config.memory_warmup_size = 1000
        tester = TestRL()
        tester.simple_check(rl_config)
