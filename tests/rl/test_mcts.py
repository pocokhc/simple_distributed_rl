import unittest

import srl
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_sequence(self):
        self.tester.play_sequence(srl.rl.mcts.Config())

    def test_mp(self):
        self.tester.play_mp(srl.rl.mcts.Config())

    def test_verify_grid(self):
        rl_config = srl.rl.mcts.Config(simulation_times=10, gamma=0.9)
        self.tester.play_verify_singleplay("Grid", rl_config, 50000)

    def test_verify_StoneTaking(self):
        rl_config = srl.rl.mcts.Config(simulation_times=10)
        self.tester.play_verify_2play("StoneTaking", rl_config, 1000)

    def test_verify_ox(self):
        rl_config = srl.rl.mcts.Config(simulation_times=10)
        self.tester.play_verify_2play("OX", rl_config, 10000, is_self_play=False)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_mp", verbosity=2)
