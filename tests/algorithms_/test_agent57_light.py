import unittest

from srl.test import TestRL
from srl.utils.common import is_package_installed

try:
    from srl.algorithms import agent57_light
except ModuleNotFoundError:
    pass


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.pendulum_config = {
            "hidden_layer_sizes": (128, 128),
            "enable_dueling_network": False,
            "memory_name": "ReplayMemory",
            "target_model_update_interval": 100,
            "q_ext_lr": 0.001,
            "q_int_lr": 0.001,
            "actor_num": 4,
            "input_ext_reward": False,
            "input_int_reward": False,
            "input_action": False,
            "enable_intrinsic_reward": True,
        }

    def test_Pendulum(self):
        rl_config = agent57_light.Config(**self.pendulum_config)
        self.tester.verify_1player("Pendulum-v1", rl_config, 200 * 40)

    def test_Pendulum_mp(self):
        rl_config = agent57_light.Config(**self.pendulum_config)
        rl_config.memory_name = "ProportionalMemory"
        self.tester.verify_1player("Pendulum-v1", rl_config, 200 * 100, is_mp=True)

    def test_Pendulum_uvfa(self):
        rl_config = agent57_light.Config(**self.pendulum_config)
        rl_config.input_ext_reward = True
        rl_config.input_int_reward = True
        rl_config.input_action = True
        self.tester.verify_1player("Pendulum-v1", rl_config, 200 * 100)

    def test_Pendulum_memory(self):
        rl_config = agent57_light.Config(**self.pendulum_config)
        rl_config.memory_name = "ProportionalMemory"
        rl_config.memory_beta_steps = 200 * 30
        self.tester.verify_1player("Pendulum-v1", rl_config, 200 * 40)

    def test_Pendulum_dis_int(self):
        rl_config = agent57_light.Config(**self.pendulum_config)
        rl_config.enable_intrinsic_reward = False
        self.tester.verify_1player("Pendulum-v1", rl_config, 200 * 40)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_Pendulum", verbosity=2)
