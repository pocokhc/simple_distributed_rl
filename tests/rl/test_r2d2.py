import unittest

import srl
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.base_config = dict(
            lstm_units=32,
            hidden_layer_sizes=(16, 16),
            enable_dueling_network=False,
            memory_name="ReplayMemory",
            target_model_update_interval=100,
            enable_rescale=False,
            burnin=5,
            sequence_length=5,
            enable_retrace=False,
        )

    def test_sequence(self):
        rl_config = srl.rl.r2d2.Config()
        self.tester.play_sequence(rl_config)

    def test_mp(self):
        rl_config = srl.rl.r2d2.Config()
        self.tester.play_mp(rl_config)

    def test_verify_Pendulum(self):
        rl_config = srl.rl.r2d2.Config(**self.base_config)
        self.tester.play_verify_singleplay("Pendulum-v1", rl_config, 200 * 35)

    def test_verify_Pendulum_retrace(self):
        rl_config = srl.rl.r2d2.Config(**self.base_config)
        rl_config.enable_retrace = True
        self.tester.play_verify_singleplay("Pendulum-v1", rl_config, 200 * 35)

    def test_verify_Pendulum_mp(self):
        rl_config = srl.rl.r2d2.Config(**self.base_config)
        self.tester.play_verify_singleplay("Pendulum-v1", rl_config, 200 * 20, is_mp=True)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_Pendulum_mp", verbosity=2)
