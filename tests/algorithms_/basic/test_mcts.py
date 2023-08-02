import unittest

from srl.algorithms import mcts
from srl.runner.runner import Runner
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
    runner = Runner("StoneTaking", rl_config)
    runner.set_seed(1)
    runner.train(max_steps=1000)
    tester.eval_2player(runner)


def test_OX():
    tester = TestRL()
    rl_config = mcts.Config(num_simulations=10)
    runner = Runner("OX", rl_config)
    runner.set_seed(2)
    runner.train(max_steps=30000)
    tester.eval_2player(runner)


def test_Grid():
    tester = TestRL()
    rl_config = mcts.Config(num_simulations=10, discount=0.9)
    runner = Runner("Grid", rl_config)
    runner.set_seed(2)
    runner.train(max_steps=50000)
    tester.eval(runner)
