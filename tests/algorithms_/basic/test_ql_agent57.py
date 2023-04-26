import srl.envs.grid  # noqa F401
import srl.envs.oneroad  # noqa F401
from srl import runner
from srl.algorithms import ql_agent57
from srl.test import TestRL


def test_Grid():
    tester = TestRL()
    rl_config = ql_agent57.Config()
    rl_config.enable_actor = False
    rl_config.epsilon = 0.5
    config = runner.Config("Grid", rl_config, seed=2)
    parameter = tester.train_eval(config, 100_000, eval_episode=100)
    tester.verify_grid_policy(rl_config, parameter)


def test_Grid_window_length():
    tester = TestRL()
    rl_config = ql_agent57.Config()
    rl_config.enable_actor = False
    rl_config.epsilon = 0.5
    rl_config.window_length = 2
    config = runner.Config("Grid", rl_config, seed=3)
    tester.train_eval(config, 50_000, eval_episode=100)


def test_Grid_mp():
    tester = TestRL()
    rl_config = ql_agent57.Config()
    rl_config.enable_actor = False
    rl_config.epsilon = 0.5
    config = runner.Config("Grid", rl_config)
    tester.train_eval(config, 50_000, is_mp=True, eval_episode=100)


def test_OneRoad():
    tester = TestRL()
    rl_config = ql_agent57.Config()
    config = runner.Config("Grid", rl_config, seed=1)
    tester.train_eval(config, 20_000)
