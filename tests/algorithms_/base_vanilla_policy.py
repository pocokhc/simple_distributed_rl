import srl
from srl.algorithms import vanilla_policy
from srl.base.define import RLBaseTypes
from tests.algorithms_.common_long_case import CommonLongCase


class LongCase(CommonLongCase):
    def test_Grid_discrete(self):
        rl_config = vanilla_policy.Config()
        runner = srl.Runner("Grid", rl_config)
        runner.set_seed(1)
        runner.train(max_train_count=10_000)
        assert runner.evaluate_compare_to_baseline_single_player()

    def test_Grid_continuous(self):
        rl_config = vanilla_policy.Config(lr=0.01)
        rl_config.override_action_type = RLBaseTypes.CONTINUOUS
        runner = srl.Runner("Grid", rl_config)
        runner.set_seed(1)
        runner.train(max_train_count=500_000)
        assert runner.evaluate_compare_to_baseline_single_player()
