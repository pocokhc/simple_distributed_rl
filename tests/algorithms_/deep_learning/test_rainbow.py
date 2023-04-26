import pytest

from srl.utils import common

from .common_base_class import CommonBaseClass


class _BaseCase(CommonBaseClass):
    def return_rl_config(self, framework):
        from srl.algorithms import rainbow

        return rainbow.Config()

    def test_Pendulum(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        rl_config.hidden_layer_sizes = (64, 64)
        rl_config.memory_beta_initial = 1.0
        tester.train_eval(config, 200 * 70)

    def test_Pendulum_mp(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        rl_config.hidden_layer_sizes = (64, 64)
        rl_config.memory_beta_initial = 1.0
        tester.train_eval(config, 200 * 70, is_mp=True)

    def test_OX(self):
        # invalid action test
        config, rl_config, tester = self.create_config("OX")
        rl_config.hidden_layer_sizes = (128,)
        rl_config.epsilon = 0.5
        rl_config.memory_name = "ReplayMemory"
        config.players = [None, "random"]
        parameter = tester.train(config, 15000)

        config.players = [None, "random"]
        tester.eval(config, parameter, baseline=[0.8, None])
        config.players = ["random", None]
        tester.eval(config, parameter, baseline=[None, 0.65])

    def _set_pendulum_config(self, rl_config) -> None:
        rl_config.epsilon = 0.1
        rl_config.discount = 0.9
        rl_config.lr = 0.001
        rl_config.batch_size = 32
        rl_config.hidden_layer_sizes = (64, 64)
        rl_config.enable_double_dqn = False
        rl_config.enable_dueling_network = False
        rl_config.enable_noisy_dense = False
        rl_config.multisteps = 1
        rl_config.memory_name = "ReplayMemory"
        rl_config.enable_rescale = False
        rl_config.window_length = 1

    def test_Pendulum_naive(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        self._set_pendulum_config(rl_config)
        tester.train_eval(config, 200 * 100)

    def test_Pendulum_window_length(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        self._set_pendulum_config(rl_config)
        rl_config.window_length = 4
        tester.train_eval(config, 200 * 70)

    def test_Pendulum_ddqn(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        self._set_pendulum_config(rl_config)
        rl_config.enable_double_dqn = True
        tester.train_eval(config, 200 * 80)

    def test_Pendulum_dueling(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        self._set_pendulum_config(rl_config)
        rl_config.enable_dueling_network = True
        tester.train_eval(config, 200 * 70)

    def test_Pendulum_noisy(self):
        pytest.importorskip("tensorflow_addons", minversion="0.17.1")

        config, rl_config, tester = self.create_config("Pendulum-v1")
        self._set_pendulum_config(rl_config)
        rl_config.enable_noisy_dense = True
        tester.train_eval(config, 200 * 80)

    def test_Pendulum_multistep(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        self._set_pendulum_config(rl_config)
        rl_config.multisteps = 10
        tester.train_eval(config, 200 * 80)

    def test_Pendulum_proportional(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        self._set_pendulum_config(rl_config)
        rl_config.memory_name = "ProportionalMemory"
        rl_config.memory_alpha = 1.0
        rl_config.memory_beta_initial = 1.0
        tester.train_eval(config, 200 * 120)

    def test_Pendulum_rankbase(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        self._set_pendulum_config(rl_config)
        rl_config.memory_name = "RankBaseMemory"
        rl_config.memory_alpha = 1.0
        rl_config.memory_beta_initial = 1.0
        tester.train_eval(config, 200 * 120)

    def test_Pendulum_rankbaseLinear(self):
        config, rl_config, tester = self.create_config("Pendulum-v1")
        self._set_pendulum_config(rl_config)
        rl_config.memory_name = "RankBaseMemoryLinear"
        rl_config.memory_alpha = 1.0
        rl_config.memory_beta_initial = 1.0
        tester.train_eval(config, 200 * 120)

    def test_Pendulum_all(self):
        pytest.importorskip("tensorflow_addons", minversion="0.17.1")

        config, rl_config, tester = self.create_config("Pendulum-v1")
        self._set_pendulum_config(rl_config)
        rl_config.enable_double_dqn = True
        rl_config.batch_size = 8
        rl_config.enable_dueling_network = True
        rl_config.enable_noisy_dense = True
        rl_config.multisteps = 5
        rl_config.memory_name = "ProportionalMemory"
        rl_config.memory_alpha = 1.0
        rl_config.memory_beta_initial = 1.0
        tester.train_eval(config, 200 * 100)


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
