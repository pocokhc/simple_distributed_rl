import unittest

import srl
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_sequence(self):
        self.tester.play_sequence(srl.rl.alphazero.Config())

    def test_mp(self):
        self.tester.play_mp(srl.rl.alphazero.Config())

    def test_verify_grid(self):
        rl_config = srl.rl.alphazero.Config()
        rl_config.simulation_times = 100
        rl_config.sampling_steps = 1
        rl_config.batch_size = 32
        rl_config.gamma = 1.0
        rl_config.lr_schedule = [
            {"train": 0, "lr": 0.02},
            {"train": 100, "lr": 0.002},
            {"train": 500, "lr": 0.0002},
        ]
        rl_config.cnn_block_kwargs = dict(n_blocks=1, filters=32)
        rl_config.value_block_kwargs = dict(hidden_layer_sizes=(32,))
        self.tester.play_verify_singleplay("Grid", rl_config, 3000)

    def test_verify_StoneTaking(self):
        rl_config = srl.rl.alphazero.Config()
        rl_config.simulation_times = 100
        rl_config.sampling_steps = 1
        rl_config.batch_size = 32
        rl_config.gamma = 1.0
        rl_config.lr_schedule = [{"train": 0, "lr": 0.02}, {"train": 100, "lr": 0.002}]
        rl_config.cnn_block_kwargs = dict(n_blocks=1, filters=32)
        rl_config.value_block_kwargs = dict(hidden_layer_sizes=(32,))
        self.tester.play_verify_2play("StoneTaking", rl_config, 300)

    def test_verify_ox(self):
        rl_config = srl.rl.alphazero.Config()
        rl_config.simulation_times = 100
        rl_config.sampling_steps = 1
        rl_config.batch_size = 32
        rl_config.gamma = 1.0
        rl_config.lr_schedule = [{"train": 0, "lr": 0.02}, {"train": 100, "lr": 0.002}]
        rl_config.cnn_block_kwargs = dict(n_blocks=1, filters=32)
        rl_config.value_block_kwargs = dict(hidden_layer_sizes=(32,))
        self.tester.play_verify_2play("OX", rl_config, 200)

    def test_verify_ox_mp(self):
        rl_config = srl.rl.alphazero.Config()
        rl_config.simulation_times = 100
        rl_config.sampling_steps = 1
        rl_config.batch_size = 32
        rl_config.gamma = 1.0
        rl_config.lr_schedule = [{"train": 0, "lr": 0.02}, {"train": 100, "lr": 0.002}]
        rl_config.cnn_block_kwargs = dict(n_blocks=1, filters=32)
        rl_config.value_block_kwargs = dict(hidden_layer_sizes=(32,))
        self.tester.play_verify_2play("OX", rl_config, 200, is_mp=True)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_ox_mp", verbosity=2)
