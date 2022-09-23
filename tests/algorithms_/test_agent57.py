import unittest

from srl.test import TestRL
from srl.utils.common import is_package_installed

try:
    from srl.algorithms import agent57
except ModuleNotFoundError:
    pass


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.base_config = dict(
            lstm_units=128,
            hidden_layer_sizes=(128,),
            enable_dueling_network=False,
            memory_name="ReplayMemory",
            target_model_update_interval=100,
            enable_rescale=True,
            q_ext_lr=0.001,
            q_int_lr=0.001,
            batch_size=32,
            burnin=5,
            sequence_length=10,
            enable_retrace=False,
            actor_num=8,
            input_ext_reward=False,
            input_int_reward=False,
            input_action=False,
            enable_intrinsic_reward=True,
        )

    def test_simple_check(self):
        self.tester.simple_check(agent57.Config())

    def test_simple_check_mp(self):
        self.tester.simple_check_mp(agent57.Config())

    def test_Pendulum(self):
        rl_config = agent57.Config(**self.base_config)
        self.tester.verify_singleplay("Pendulum-v1", rl_config, 200 * 50)

    def test_Pendulum_mp(self):
        rl_config = agent57.Config(**self.base_config)
        self.tester.verify_singleplay("Pendulum-v1", rl_config, 200 * 40, is_mp=True)

    def test_Pendulum_retrace(self):
        rl_config = agent57.Config(**self.base_config)
        rl_config.enable_retrace = True
        self.tester.verify_singleplay("Pendulum-v1", rl_config, 200 * 50)

    def test_Pendulum_uvfa(self):
        rl_config = agent57.Config(**self.base_config)
        rl_config.input_ext_reward = True
        rl_config.input_int_reward = True
        rl_config.input_action = True
        self.tester.verify_singleplay("Pendulum-v1", rl_config, 200 * 150)

    def test_Pendulum_memory(self):
        rl_config = agent57.Config(**self.base_config)
        rl_config.memory_name = "ProportionalMemory"
        rl_config.memory_beta_steps = 200 * 30
        self.tester.verify_singleplay("Pendulum-v1", rl_config, 200 * 60)

    def test_Pendulum_dis_int(self):
        rl_config = agent57.Config(**self.base_config)
        rl_config.enable_intrinsic_reward = False
        self.tester.verify_singleplay("Pendulum-v1", rl_config, 200 * 50)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_simple_check", verbosity=2)
