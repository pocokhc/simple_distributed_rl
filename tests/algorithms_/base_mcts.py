import srl
from srl.algorithms import mcts
from tests.algorithms_.common_long_case import CommonLongCase


class LongCase(CommonLongCase):
    def test_StoneTaking(self):
        rl_config = mcts.Config(num_simulations=10)
        runner = srl.Runner("StoneTaking", rl_config)
        runner.set_seed(1)
        runner.train(max_steps=1000)
        results = runner.evaluate_compare_to_baseline_multiplayer()
        assert all(results)

    def test_OX(self):
        rl_config = mcts.Config(num_simulations=10)
        runner = srl.Runner("OX", rl_config)
        runner.set_seed(2)
        runner.train(max_steps=30000)
        results = runner.evaluate_compare_to_baseline_multiplayer()
        assert all(results)

    def test_Grid(self):
        rl_config = mcts.Config(num_simulations=10, discount=0.9)
        runner = srl.Runner("Grid", rl_config)
        runner.set_seed(2)
        runner.train(max_steps=10000)
        assert runner.evaluate_compare_to_baseline_single_player()
