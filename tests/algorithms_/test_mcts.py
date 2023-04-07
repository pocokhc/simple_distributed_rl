import unittest

import srl.envs.grid  # noqa F401
import srl.envs.ox  # noqa F401
import srl.envs.stone_taking  # noqa F401
from srl.algorithms import mcts
from srl.test import TestRL


class Test(TestRL):
    def setUp(self) -> None:
        self.rl_config = mcts.Config()
        self.simple_check_kwargs = {
            "train_kwargs": {
                "max_steps": 10,
            }
        }

    def test_Grid(self):
        rl_config = mcts.Config(num_simulations=10, discount=0.9)
        self.verify_1player("Grid", rl_config, train_count=-1, train_steps=60000)

    def test_StoneTaking(self):
        rl_config = mcts.Config(num_simulations=10)
        self.verify_2player("StoneTaking", rl_config, train_count=-1, train_steps=1000)

    def test_OX(self):
        rl_config = mcts.Config(num_simulations=10)
        self.verify_2player("OX", rl_config, train_count=-1, train_steps=10000)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_StoneTaking", verbosity=2)
