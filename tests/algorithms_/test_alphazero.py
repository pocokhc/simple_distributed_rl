import unittest

from srl.test import TestRL
from srl.utils.common import is_package_installed

try:
    import srl.envs.grid  # noqa F401
    import srl.envs.othello  # noqa F401
    import srl.envs.ox  # noqa F401
    import srl.envs.stone_taking  # noqa F401
    from srl.algorithms import alphazero
except ModuleNotFoundError:
    pass


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.rl_config = alphazero.Config()

    def test_Grid(self):
        self.rl_config.set_parameter(
            dict(
                num_simulations=100,
                sampling_steps=1,
                batch_size=64,
                warmup_size=100,
                discount=0.9,
                lr_schedule=[
                    {"train": 0, "lr": 0.02},
                    {"train": 100, "lr": 0.002},
                    {"train": 500, "lr": 0.0002},
                ],
                input_image_block_kwargs=dict(n_blocks=1, filters=32),
                value_block_kwargs=dict(layer_sizes=(32,)),
            )
        )
        self.tester.verify_1player("Grid", self.rl_config, 5000)

    def test_StoneTaking(self):
        rl_config = alphazero.Config()
        rl_config.num_simulations = 100
        rl_config.sampling_steps = 1
        rl_config.batch_size = 32
        rl_config.discount = 1.0
        rl_config.lr_schedule = [{"train": 0, "lr": 0.02}, {"train": 100, "lr": 0.002}]
        rl_config.input_image_block_kwargs = dict(n_blocks=1, filters=32)
        rl_config.value_block_kwargs = dict(layer_sizes=(32,))
        self.tester.verify_2player("StoneTaking", rl_config, 300)

    def test_OX(self):
        rl_config = alphazero.Config()
        rl_config.num_simulations = 100
        rl_config.sampling_steps = 1
        rl_config.batch_size = 32
        rl_config.discount = 1.0
        rl_config.lr_schedule = [{"train": 0, "lr": 0.02}, {"train": 100, "lr": 0.002}]
        rl_config.input_image_block_kwargs = dict(n_blocks=1, filters=32)
        rl_config.value_block_kwargs = dict(layer_sizes=(32,))
        self.tester.verify_2player("OX", rl_config, 200)

    def test_OX_mp(self):
        rl_config = alphazero.Config()
        rl_config.num_simulations = 100
        rl_config.sampling_steps = 1
        rl_config.batch_size = 32
        rl_config.discount = 1.0
        rl_config.lr_schedule = [{"train": 0, "lr": 0.02}, {"train": 100, "lr": 0.002}]
        rl_config.input_image_block_kwargs = dict(n_blocks=1, filters=32)
        rl_config.value_block_kwargs = dict(layer_sizes=(32,))
        self.tester.verify_2player("OX", rl_config, 200, is_mp=True)


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class TestLong(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_Othello4x4(self):
        rl_config = alphazero.Config()
        rl_config.num_simulations = 100
        rl_config.sampling_steps = 1
        rl_config.batch_size = 128
        rl_config.warmup_size = 500
        rl_config.capacity = 100_000
        rl_config.lr_schedule = [
            {"train": 0, "lr": 0.001},
            {"train": 1000, "lr": 0.0005},
            {"train": 5000, "lr": 0.0002},
        ]
        rl_config.input_image_block_kwargs = dict(n_blocks=19, filters=128)
        rl_config.value_block_kwargs = dict(layer_sizes=(128,))
        rl_config.policy_block_kwargs = dict(layer_sizes=(128,))
        self.tester.verify_2player("Othello4x4", rl_config, 400000, is_mp=True)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_grid", verbosity=2)
