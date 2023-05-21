from srl import runner
from srl.algorithms import vanilla_policy
from srl.base.define import RLActionTypes
from srl.test import TestRL


def test_Grid_discrete():
    tester = TestRL()
    rl_config = vanilla_policy.Config()
    config = runner.Config("Grid", rl_config, seed=1)
    tester.train_eval(config, 10_000, eval_episode=100)


def test_Grid_continuous():
    tester = TestRL()
    rl_config = vanilla_policy.Config()
    rl_config.override_rl_action_type = RLActionTypes.CONTINUOUS
    config = runner.Config("Grid", rl_config, seed=1)
    tester.train_eval(config, 150_000, eval_episode=100)
