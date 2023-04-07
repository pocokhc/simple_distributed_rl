import unittest

from srl.test import TestRL
from srl.utils.common import is_package_installed

try:
    from srl.algorithms import r2d2
except ModuleNotFoundError:
    pass


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.base_config = dict(
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
        rl_config = r2d2.Config(**self.base_config)
        self.tester.verify_1player("Pendulum-v1", rl_config, 200 * 35)

    def test_Pendulum_mp(self):
        rl_config = r2d2.Config(**self.base_config)
        self.tester.verify_1player("Pendulum-v1", rl_config, 200 * 20, is_mp=True)

    def test_Pendulum_retrace(self):
        rl_config = r2d2.Config(**self.base_config)
        rl_config.enable_retrace = True
        self.tester.verify_1player("Pendulum-v1", rl_config, 200 * 35)

    def test_Pendulum_memory(self):
        rl_config = r2d2.Config(**self.base_config)
        rl_config.memory_name = "ProportionalMemory"
        self.tester.verify_1player("Pendulum-v1", rl_config, 200 * 50)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_Pendulum", verbosity=2)
