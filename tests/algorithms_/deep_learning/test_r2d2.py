import pytest

from srl.utils import common

from .common_base_class import CommonBaseClass


class _BaseCase(CommonBaseClass):
    def return_rl_config(self, framework):
        from srl.algorithms import r2d2

        return r2d2.Config(
            lstm_units=32,
            hidden_layer_sizes=(16, 16),
            enable_dueling_network=False,
            memory_name="ReplayMemory",
            target_model_update_interval=100,
            enable_rescale=True,
            burnin=5,
            sequence_length=5,
            enable_retrace=False,
        )

    def test_Pendulum(self):
        config, _, tester = self.create_config("Pendulum-v1")
        tester.train_eval(config, 200 * 35)

    def test_Pendulum_mp(self):
        config, _, tester = self.create_config("Pendulum-v1")
        tester.train_eval(config, 200 * 20, is_mp=True)

    def test_Pendulum_retrace(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        rl_config.enable_retrace = True
        tester.train_eval(config, 200 * 35)

    def test_Pendulum_memory(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        rl_config.memory_name = "ProportionalMemory"
        tester.train_eval(config, 200 * 50)


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
