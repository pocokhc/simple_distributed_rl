import unittest

import srl
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.rl_config = srl.rl.rainbow.Config()

    def test_sequence(self):
        self.tester.play_sequence(self.rl_config)

    def test_mp(self):
        self.tester.play_mp(self.rl_config)

    def test_verify_Grid(self):
        self.rl_config.epsilon = 0.5
        self.rl_config.lr = 0.001
        self.rl_config.memory_name = "ReplayMemory"
        self.rl_config.memory_warmup_size = 100
        self.rl_config.hidden_layer_sizes = (32, 32, 32)
        self.rl_config.enable_rescale = False
        self.tester.play_verify_singleplay("Grid", self.rl_config, 5000)

    def test_verify_Pendulum(self):
        self.rl_config.hidden_layer_sizes = (64, 64)
        self.rl_config.memory_beta_initial = 1.0
        self.tester.play_verify_singleplay("Pendulum-v1", self.rl_config, 200 * 70)

    def test_verify_Pendulum_mp(self):
        self.rl_config.hidden_layer_sizes = (64, 64)
        self.rl_config.memory_beta_initial = 1.0
        self.tester.play_verify_singleplay("Pendulum-v1", self.rl_config, 200 * 70, is_mp=True)

    def test_verify_OX(self):
        # invalid action test
        self.rl_config.hidden_layer_sizes = (128,)
        self.rl_config.epsilon = 0.5
        self.rl_config.memory_name = "ReplayMemory"
        self.tester.play_verify_2play("OX", self.rl_config, 15000)


class TestPendulum(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.rl_config = srl.rl.rainbow.Config(
            epsilon=0.1,
            gamma=0.9,
            lr=0.001,
            batch_size=32,
            hidden_layer_sizes=(64, 64),
            window_length=1,
            enable_double_dqn=False,
            enable_dueling_network=False,
            enable_noisy_dense=False,
            multisteps=1,
            memory_name="ReplayMemory",
            enable_rescale=False,
        )

    def test_verify_naive(self):
        self.tester.play_verify_singleplay("Pendulum-v1", self.rl_config, 200 * 100)

    def test_verify_ddqn(self):
        self.rl_config.enable_double_dqn = True
        self.tester.play_verify_singleplay("Pendulum-v1", self.rl_config, 200 * 80)

    def test_verify_dueling(self):
        self.rl_config.enable_dueling_network = True
        self.tester.play_verify_singleplay("Pendulum-v1", self.rl_config, 200 * 70)

    def test_verify_noisy(self):
        self.rl_config.enable_noisy_dense = True
        self.tester.play_verify_singleplay("Pendulum-v1", self.rl_config, 200 * 80)

    def test_verify_multistep(self):
        self.rl_config.multisteps = 10
        self.tester.play_verify_singleplay("Pendulum-v1", self.rl_config, 200 * 70)

    def test_verify_proportional(self):
        self.rl_config.memory_name = "ProportionalMemory"
        self.rl_config.memory_alpha = 1.0
        self.rl_config.memory_beta_initial = 1.0
        self.tester.play_verify_singleplay("Pendulum-v1", self.rl_config, 200 * 120)

    def test_verify_rankbase(self):
        self.rl_config.memory_name = "RankBaseMemory"
        self.rl_config.memory_alpha = 1.0
        self.rl_config.memory_beta_initial = 1.0
        self.tester.play_verify_singleplay("Pendulum-v1", self.rl_config, 200 * 120)

    def test_verify_rankbaseLinear(self):
        self.rl_config.memory_name = "RankBaseMemoryLinear"
        self.rl_config.memory_alpha = 1.0
        self.rl_config.memory_beta_initial = 1.0
        self.tester.play_verify_singleplay("Pendulum-v1", self.rl_config, 200 * 120)

    def test_verify_all(self):
        self.rl_config.enable_double_dqn = True
        self.rl_config.lr = 0.001
        self.rl_config.batch_size = 8
        self.rl_config.enable_dueling_network = True
        # self.rl_config.enable_noisy_dense = True
        self.rl_config.multisteps = 5
        self.rl_config.memory_name = "ProportionalMemory"
        self.rl_config.memory_alpha = 1.0
        self.rl_config.memory_beta_initial = 1.0
        self.tester.play_verify_singleplay("Pendulum-v1", self.rl_config, 200 * 100)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_Pendulum_mp", verbosity=2)
    # unittest.main(module=__name__, defaultTest="TestPendulum.test_verify_dueling", verbosity=2)
