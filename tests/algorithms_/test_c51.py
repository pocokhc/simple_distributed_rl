import unittest

from srl.test import TestRL
from srl.utils.common import is_package_installed

try:
    import srl.envs.grid  # noqa F401
    from srl.algorithms import c51
except ModuleNotFoundError:
    pass


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.rl_config = c51.Config()

    def test_Grid(self):
        self.rl_config.epsilon = 0.5
        self.rl_config.lr = 0.002
        self.rl_config.hidden_block_kwargs = dict(hidden_layer_sizes=(16, 16))
        self.rl_config.categorical_num_atoms = 11
        self.rl_config.categorical_v_min = -2
        self.rl_config.categorical_v_max = 2
        self.tester.verify_1player("Grid", self.rl_config, 6000)

    def test_Pendulum(self):
        self.rl_config.categorical_v_min = -100
        self.rl_config.categorical_v_max = 100
        self.rl_config.batch_size = 64
        self.rl_config.lr = 0.001
        self.rl_config.hidden_block_kwargs = dict(hidden_layer_sizes=(32, 32, 32))
        self.tester.verify_1player("Pendulum-v1", self.rl_config, 200 * 600)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_Pendulum", verbosity=2)
