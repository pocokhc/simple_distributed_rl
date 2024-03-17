from typing import Tuple

import srl
from srl.algorithms import mcts
from srl.base.rl.config import RLConfig
from srl.test.rl import TestRL
from tests.algorithms_.common_quick_case import CommonQuickCase


class QuickCase(CommonQuickCase):
    def create_rl_config(self, rl_param) -> Tuple[RLConfig, dict]:
        rl_config = mcts.Config(num_simulations=1)
        return rl_config, dict(
            use_layer_processor=True,
            train_kwargs=dict(max_steps=2, timeout=1),
        )


class BaseCase:
    def test_StoneTaking(self):
        tester = TestRL()
        rl_config = mcts.Config(num_simulations=10)
        runner = srl.Runner("StoneTaking", rl_config)
        runner.set_seed(1)
        runner.train(max_steps=1000)
        tester.eval_2player(runner)

    def test_OX(self):
        tester = TestRL()
        rl_config = mcts.Config(num_simulations=10)
        runner = srl.Runner("OX", rl_config)
        runner.set_seed(2)
        runner.train(max_steps=30000)
        tester.eval_2player(runner)

    def test_Grid(self):
        tester = TestRL()
        rl_config = mcts.Config(num_simulations=10, discount=0.9)
        runner = srl.Runner("Grid", rl_config)
        runner.set_seed(2)
        runner.train(max_steps=50000)
        tester.eval(runner)
