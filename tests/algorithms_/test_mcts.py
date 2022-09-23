import unittest

import srl.envs.grid  # noqa F401
import srl.envs.ox  # noqa F401
import srl.envs.stone_taking  # noqa F401
from srl.algorithms import mcts
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_simple_check(self):
        self.tester.simple_check(mcts.Config())

    def test_simple_check_mp(self):
        self.tester.simple_check_mp(mcts.Config())

    def test_verify_grid(self):
        rl_config = mcts.Config(num_simulations=10, discount=0.9)
        self.tester.verify_singleplay("Grid", rl_config, 50000)

    def test_verify_StoneTaking(self):
        rl_config = mcts.Config(num_simulations=10)
        self.tester.verify_2play("StoneTaking", rl_config, 1000)

    def test_verify_ox(self):
        rl_config = mcts.Config(num_simulations=10)
        self.tester.verify_2play("OX", rl_config, 10000, is_self_play=False)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_ox", verbosity=2)
