import srl
from srl.algorithms import ql_agent57
from tests.algorithms_.common_long_case import CommonLongCase


class LongCase(CommonLongCase):
    def test_Grid(self):
        rl_config = ql_agent57.Config()
        rl_config.enable_actor = False
        rl_config.epsilon = 0.5
        runner = srl.Runner("Grid", rl_config)
        runner.set_seed(3)
        runner.train(max_train_count=100_000)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Grid_window_length(self):
        rl_config = ql_agent57.Config()
        rl_config.enable_actor = False
        rl_config.epsilon = 0.5
        rl_config.window_length = 2
        runner = srl.Runner("Grid", rl_config)
        runner.set_seed(3)
        runner.train(max_train_count=50_000)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Grid_mp(self):
        rl_config = ql_agent57.Config()
        rl_config.enable_actor = False
        rl_config.epsilon = 0.5
        runner = srl.Runner("Grid", rl_config)
        runner.train_mp(max_train_count=50_000)
        assert runner.evaluate_compare_to_baseline_single_player()
