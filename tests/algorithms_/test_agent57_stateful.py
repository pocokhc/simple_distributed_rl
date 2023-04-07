import unittest

from srl.test import TestRL
from srl.utils.common import is_package_installed

try:
    from srl.algorithms import agent57_stateful
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

    def test_Pendulum(self):
        rl_config = agent57_stateful.Config(**self.base_config)
        self.tester.verify_1player("Pendulum-v1", rl_config, 200 * 50)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_Pendulum", verbosity=2)
