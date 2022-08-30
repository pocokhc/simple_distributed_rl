import unittest

from algorithms import c51
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.rl_config = c51.Config()

    def test_simple_check(self):
        self.tester.simple_check(c51.Config())

    def test_simple_check_mp(self):
        self.tester.simple_check_mp(c51.Config())

    def test_verify_Grid(self):
        self.rl_config.epsilon = 0.5
        self.rl_config.lr = 0.002
        self.rl_config.hidden_block_kwargs = dict(hidden_layer_sizes=(16, 16))
        self.rl_config.categorical_num_atoms = 11
        self.rl_config.categorical_v_min = -2
        self.rl_config.categorical_v_max = 2
        self.tester.verify_singleplay("Grid", self.rl_config, 6000)

    def test_verify_Pendulum(self):
        self.rl_config.categorical_v_min = -100
        self.rl_config.categorical_v_max = 100
        self.rl_config.batch_size = 64
        self.rl_config.lr = 0.001
        self.rl_config.hidden_block_kwargs = dict(hidden_layer_sizes=(32, 32, 32))
        self.tester.verify_singleplay("Pendulum-v1", self.rl_config, 200 * 600)


if __name__ == "__main__":
    import __init__  # noqa F401

    unittest.main(module=__name__, defaultTest="Test.test_verify_Pendulum", verbosity=2)
