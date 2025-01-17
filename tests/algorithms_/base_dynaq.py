import srl
from srl.algorithms import dynaq
from tests.algorithms_.common_long_case import CommonLongCase


class LongCase(CommonLongCase):
    def test_Grid(self):
        rl_config = dynaq.Config()
        runner = srl.Runner("Grid", rl_config)
        runner.set_seed(1)
        runner.train(max_train_count=50_000)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Grid_mp(self):
        rl_config = dynaq.Config()
        runner = srl.Runner("Grid", rl_config)
        runner.set_seed(5)
        runner.train_mp(max_train_count=50_000)
        assert runner.evaluate_compare_to_baseline_single_player()
