import pytest

from srl.rl.memories.config import (
    ProportionalMemoryConfig,
    RankBaseMemoryConfig,
    RankBaseMemoryLinearConfig,
    ReplayMemoryConfig,
)

from .common_base_class import CommonBaseClass


class BaseCase(CommonBaseClass):
    def _create_rl_config(self):
        from srl.algorithms import rainbow

        return rainbow.Config()

    def test_Pendulum(self):
        rl_config = self._create_rl_config()
        rl_config.hidden_layer_sizes = (64, 64)
        rl_config.multisteps = 3
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 70)

    def test_Pendulum_mp(self):
        rl_config = self._create_rl_config()
        rl_config.hidden_layer_sizes = (64, 64)
        rl_config.multisteps = 3
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 70, is_mp=True)

    def test_Pendulum_noisy(self):
        pytest.importorskip("tensorflow_addons", minversion="0.17.1")

        rl_config = self._create_rl_config()
        rl_config.hidden_layer_sizes = (64, 64)
        rl_config.multisteps = 3
        rl_config.enable_noisy_dense = True
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 70)

    def test_Pendulum_no_multi(self):
        rl_config = self._create_rl_config()
        rl_config.hidden_layer_sizes = (64, 64)
        rl_config.multisteps = 1
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 70)

    def test_Pendulum_no_multi_mp(self):
        rl_config = self._create_rl_config()
        rl_config.hidden_layer_sizes = (64, 64)
        rl_config.multisteps = 1
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 70, is_mp=True)

    def test_Pendulum_no_multi_noisy(self):
        pytest.importorskip("tensorflow_addons", minversion="0.17.1")

        rl_config = self._create_rl_config()
        rl_config.hidden_layer_sizes = (64, 64)
        rl_config.multisteps = 1
        rl_config.enable_noisy_dense = True
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 70)

    def test_OX(self):
        # invalid action test
        rl_config = self._create_rl_config()
        rl_config.hidden_layer_sizes = (128,)
        rl_config.epsilon = 0.5
        rl_config.memory = ReplayMemoryConfig()

        config, tester = self.create_config("OX", rl_config)
        config.players = [None, "random"]
        parameter, _, _ = tester.train(config, 10000)

        config.players = [None, "random"]
        tester.eval(config, parameter, baseline=[0.8, None])
        config.players = ["random", None]
        tester.eval(config, parameter, baseline=[None, 0.65])

    def _create_pendulum_config(self):
        rl_config = self._create_rl_config()
        rl_config.epsilon = 0.1
        rl_config.discount = 0.9
        rl_config.lr = 0.001
        rl_config.batch_size = 32
        rl_config.hidden_layer_sizes = (64, 64)
        rl_config.enable_double_dqn = False
        rl_config.enable_dueling_network = False
        rl_config.enable_noisy_dense = False
        rl_config.multisteps = 1
        rl_config.memory = ReplayMemoryConfig()
        rl_config.enable_rescale = False
        rl_config.window_length = 1
        return rl_config

    def test_Pendulum_naive(self):
        rl_config = self._create_pendulum_config()
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 100)

    def test_Pendulum_window_length(self):
        rl_config = self._create_pendulum_config()
        rl_config.window_length = 4
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 70)

    def test_Pendulum_ddqn(self):
        rl_config = self._create_pendulum_config()
        rl_config.enable_double_dqn = True
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 80)

    def test_Pendulum_dueling(self):
        rl_config = self._create_pendulum_config()
        rl_config.enable_dueling_network = True
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 70)

    def test_Pendulum_multistep(self):
        rl_config = self._create_pendulum_config()
        rl_config.multisteps = 10
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 80)

    def test_Pendulum_proportional(self):
        rl_config = self._create_pendulum_config()
        rl_config.memory = ProportionalMemoryConfig(alpha=1.0, beta_initial=1.0)
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 120)

    def test_Pendulum_rankbase(self):
        rl_config = self._create_pendulum_config()
        rl_config.memory = RankBaseMemoryConfig(alpha=1.0, beta_initial=1.0)
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 120)

    def test_Pendulum_rankbaseLinear(self):
        rl_config = self._create_pendulum_config()
        rl_config.memory = RankBaseMemoryLinearConfig(alpha=1.0, beta_initial=1.0)
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 120)

    def test_Pendulum_all(self):
        pytest.importorskip("tensorflow_addons", minversion="0.17.1")

        rl_config = self._create_pendulum_config()
        rl_config.enable_double_dqn = True
        rl_config.batch_size = 8
        rl_config.enable_dueling_network = True
        rl_config.enable_noisy_dense = True
        rl_config.multisteps = 5
        rl_config.memory = ProportionalMemoryConfig(alpha=1.0, beta_initial=1.0)
        config, tester = self.create_config("Pendulum-v1", rl_config)
        tester.train_eval(config, 200 * 100)
