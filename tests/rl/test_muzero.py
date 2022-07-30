import unittest

import srl
from srl.rl.muzero import _category_decode, _encode_category
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_category(self):
        # マイナス
        cat = _encode_category(-2.6, -5, 5)
        self.assertAlmostEqual(cat[2], 0.6)
        self.assertAlmostEqual(cat[3], 0.4)

        val = _category_decode(cat, -5)
        self.assertAlmostEqual(val, -2.6)

        # plus
        cat = _encode_category(2.4, -5, 5)
        self.assertAlmostEqual(cat[7], 0.6)
        self.assertAlmostEqual(cat[8], 0.4)

        val = _category_decode(cat, -5)
        self.assertAlmostEqual(val, 2.4)

    def test_sequence(self):
        self.tester.play_sequence(srl.rl.alphazero.Config())

    def test_mp(self):
        self.tester.play_mp(srl.rl.alphazero.Config())

    def test_verify_grid(self):
        rl_config = srl.rl.alphazero.Config()
        rl_config.simulation_times = 100
        rl_config.sampling_steps = 1
        rl_config.batch_size = 64
        rl_config.warmup_size = 100
        rl_config.discount = 0.9
        rl_config.lr_schedule = [
            {"train": 0, "lr": 0.02},
            {"train": 100, "lr": 0.002},
            {"train": 500, "lr": 0.0002},
        ]
        rl_config.cnn_block_kwargs = dict(n_blocks=1, filters=32)
        rl_config.value_block_kwargs = dict(hidden_layer_sizes=(32,))
        self.tester.play_verify_singleplay("Grid", rl_config, 5000)

    def test_verify_StoneTaking(self):
        rl_config = srl.rl.alphazero.Config()
        rl_config.simulation_times = 100
        rl_config.sampling_steps = 1
        rl_config.batch_size = 32
        rl_config.discount = 1.0
        rl_config.lr_schedule = [{"train": 0, "lr": 0.02}, {"train": 100, "lr": 0.002}]
        rl_config.cnn_block_kwargs = dict(n_blocks=1, filters=32)
        rl_config.value_block_kwargs = dict(hidden_layer_sizes=(32,))
        self.tester.play_verify_2play("StoneTaking", rl_config, 300)

    def test_verify_ox(self):
        rl_config = srl.rl.alphazero.Config()
        rl_config.simulation_times = 100
        rl_config.sampling_steps = 1
        rl_config.batch_size = 32
        rl_config.discount = 1.0
        rl_config.lr_schedule = [{"train": 0, "lr": 0.02}, {"train": 100, "lr": 0.002}]
        rl_config.cnn_block_kwargs = dict(n_blocks=1, filters=32)
        rl_config.value_block_kwargs = dict(hidden_layer_sizes=(32,))
        self.tester.play_verify_2play("OX", rl_config, 200)

    def test_verify_ox_mp(self):
        rl_config = srl.rl.alphazero.Config()
        rl_config.simulation_times = 100
        rl_config.sampling_steps = 1
        rl_config.batch_size = 32
        rl_config.discount = 1.0
        rl_config.lr_schedule = [{"train": 0, "lr": 0.02}, {"train": 100, "lr": 0.002}]
        rl_config.cnn_block_kwargs = dict(n_blocks=1, filters=32)
        rl_config.value_block_kwargs = dict(hidden_layer_sizes=(32,))
        self.tester.play_verify_2play("OX", rl_config, 200, is_mp=True)

    def test_verify_Othello4x4(self):
        rl_config = srl.rl.alphazero.Config()
        rl_config.simulation_times = 100
        rl_config.sampling_steps = 1
        rl_config.batch_size = 128
        rl_config.warmup_size = 500
        rl_config.capacity = 100_000
        rl_config.lr_schedule = [
            {"train": 0, "lr": 0.001},
            {"train": 1000, "lr": 0.0005},
            {"train": 5000, "lr": 0.0002},
        ]
        rl_config.cnn_block_kwargs = dict(n_blocks=19, filters=128)
        rl_config.value_block_kwargs = dict(hidden_layer_sizes=(128,))
        rl_config.policy_block_kwargs = dict(hidden_layer_sizes=(128,))
        self.tester.play_verify_2play("Othello4x4", rl_config, 400000, is_mp=True)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_category", verbosity=2)
