import srl.envs.grid  # noqa F401
from srl import runner
from srl.algorithms import dynaq
from srl.test import TestRL


def test_Grid():
    tester = TestRL()
    rl_config = dynaq.Config()
    config = runner.Config("Grid", rl_config, seed=5)
    parameter, _, _ = tester.train_eval(config, 50_000)
    tester.verify_grid_policy(rl_config, parameter)


def test_Grid_mp():
    tester = TestRL()
    rl_config = dynaq.Config()
    config = runner.Config("Grid", rl_config, seed=1)
    tester.train_eval(config, 50_000, is_mp=True)
