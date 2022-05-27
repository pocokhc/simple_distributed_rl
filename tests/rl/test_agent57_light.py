import unittest

import srl
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.rl_config = srl.rl.agent57_light.Config()

    def test_sequence(self):
        self.tester.play_sequence(self.rl_config)

    def test_mp(self):
        self.tester.play_mp(self.rl_config)

    def test_verify_Pendulum(self):
        self.rl_config.hidden_layer_sizes = (32, 32, 32)
        self.rl_config.enable_dueling_network = False
        self.rl_config.memory_name = "RankBaseMemory"
        self.rl_config.memory_alpha = 0.8
        self.rl_config.memory_beta_initial = 1.0
        self.tester.play_verify_singleplay("Pendulum-v1", self.rl_config, 200 * 50, 10)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_sequence", verbosity=2)
