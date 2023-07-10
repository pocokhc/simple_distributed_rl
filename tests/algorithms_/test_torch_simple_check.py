import unittest

import pytest

from srl.test import TestRL
from srl.utils import common

common.logger_print()


class Test_agent57_light(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        pytest.importorskip("torch")

        from srl.algorithms import agent57_light

        self.rl_config = agent57_light.Config(framework="torch")
        self.rl_config.memory_warmup_size = 100


class Test_dqn(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        pytest.importorskip("torch")

        from srl.algorithms import dqn

        self.rl_config = dqn.Config(framework="torch")

    def test_simple_check_atari_config(self):
        self.init_simple_check()
        self.rl_config.use_render_image_for_observation = True
        self.rl_config.set_atari_config()
        self.rl_config.memory_warmup_size = 1000
        self.simple_check(self.rl_config)


class Test_rainbow(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        pytest.importorskip("torch")

        from srl.algorithms import rainbow

        self.rl_config = rainbow.Config(framework="torch")

    def test_simple_check_atari_config(self):
        pytest.importorskip("torch")

        self.init_simple_check()
        self.rl_config.set_atari_config()
        self.rl_config.memory_warmup_size = 1000
        self.simple_check(self.rl_config)
