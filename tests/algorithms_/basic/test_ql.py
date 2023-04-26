import srl.envs.grid  # noqa F401
import srl.envs.ox  # noqa F401
import srl.envs.tiger  # noqa F401
from srl import runner
from srl.algorithms import ql
from srl.test import TestRL
from srl.utils import common

common.logger_print()


def test_Grid():
    tester = TestRL()
    rl_config = ql.Config(
        epsilon=0.5,
        lr=0.01,
    )
    config = runner.Config("Grid", rl_config, seed=2)
    parameter = tester.train_eval(config, 100_000, eval_episode=100)
    tester.verify_grid_policy(rl_config, parameter)


def test_Grid_mp():
    tester = TestRL()
    rl_config = ql.Config(
        epsilon=0.5,
        lr=0.01,
    )
    config = runner.Config("Grid", rl_config)
    tester.train_eval(config, 200_000, is_mp=True, eval_episode=100)


def test_Grid_random():
    tester = TestRL()
    rl_config = ql.Config(
        epsilon=0.5,
        q_init="random",
    )
    config = runner.Config("Grid", rl_config, seed=2)
    tester.train_eval(config, 100_000, eval_episode=100)


def test_OX():
    tester = TestRL()
    rl_config = ql.Config(
        epsilon=0.5,
        lr=0.1,
    )
    config = runner.Config("OX", rl_config, seed=1)
    parameter = tester.train(config, 100_000)
    tester.eval_2player(config, parameter)


def test_Tiger():
    tester = TestRL()
    rl_config = ql.Config()
    rl_config.window_length = 10
    config = runner.Config("Tiger", rl_config, seed=2)
    tester.train_eval(config, 500_000)
