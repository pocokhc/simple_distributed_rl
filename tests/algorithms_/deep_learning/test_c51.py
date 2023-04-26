import pytest

from srl.rl.models.mlp_block_config import MLPBlockConfig
from srl.utils import common

from .common_base_class import CommonBaseClass


class _BaseCase(CommonBaseClass):
    def return_rl_config(self, framework):
        from srl.algorithms import c51

        return c51.Config()

    def test_Grid(self):
        config, rl_config, tester = self.create_config("Grid")
        rl_config.epsilon = 0.5
        rl_config.lr = 0.002
        rl_config.hidden_block = MLPBlockConfig(layer_sizes=(16, 16))
        rl_config.categorical_num_atoms = 11
        rl_config.categorical_v_min = -2
        rl_config.categorical_v_max = 2
        tester.train_eval(config, 6000)

    def test_Pendulum(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        rl_config.categorical_v_min = -100
        rl_config.categorical_v_max = 100
        rl_config.batch_size = 64
        rl_config.lr = 0.001
        rl_config.hidden_block = MLPBlockConfig(layer_sizes=(32, 32, 32))
        tester.train_eval(config, 200 * 600)

    def test_Pendulum_mp(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        rl_config.categorical_v_min = -100
        rl_config.categorical_v_max = 100
        rl_config.batch_size = 64
        rl_config.lr = 0.001
        rl_config.hidden_block = MLPBlockConfig(layer_sizes=(32, 32, 32))
        tester.train_eval(config, 200 * 600, is_mp=True)


class TestTF_CPU(_BaseCase):
    def return_params(self):
        pytest.importorskip("tensorflow")

        return "tensorflow", "CPU"


class TestTF_GPU(_BaseCase):
    def return_params(self):
        pytest.importorskip("tensorflow")
        if not common.is_available_gpu_tf():
            pytest.skip()

        return "tensorflow", "GPU"
