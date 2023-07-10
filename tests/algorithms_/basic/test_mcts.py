import unittest

from srl import runner
from srl.algorithms import mcts
from srl.test import TestRL


class Test_mcts(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        from srl.algorithms import mcts

        self.rl_config = mcts.Config()
        self.simple_check_kwargs = dict(
            use_layer_processor=True,
            train_kwargs=dict(max_steps=10),
        )


def test_StoneTaking():
    tester = TestRL()
    rl_config = mcts.Config(num_simulations=10)
    config = runner.Config("StoneTaking", rl_config, seed=1)
    parameter, _, _ = tester.train(config, train_steps=1000)
    tester.eval_2player(config, parameter)


def test_OX():
    tester = TestRL()
    rl_config = mcts.Config(num_simulations=10)
    config = runner.Config("OX", rl_config, seed=2)
    parameter, _, _ = tester.train(config, train_steps=30000)
    tester.eval_2player(config, parameter)


def test_Grid():
    tester = TestRL()
    rl_config = mcts.Config(num_simulations=10, discount=0.9)
    config = runner.Config("Grid", rl_config, seed=2)
    tester.train_eval(config, train_steps=50000)
