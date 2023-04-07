import unittest

import srl.envs.grid  # noqa F401
import srl.envs.ox  # noqa F401
import srl.envs.tiger  # noqa F401
from srl.algorithms import ql
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_Grid(self):
        rl_config = ql.Config(
            epsilon=0.5,
            lr=0.01,
        )
        parameter = self.tester.verify_1player("Grid", rl_config, 50_000)
        self.tester.verify_grid_policy(rl_config, parameter)

    def test_Grid_mp(self):
        rl_config = ql.Config(
            epsilon=0.5,
            lr=0.01,
        )
        parameter = self.tester.verify_1player("Grid", rl_config, 50_000, is_mp=True)
        self.tester.verify_grid_policy(rl_config, parameter)

    def test_Grid_random(self):
        rl_config = ql.Config(
            epsilon=0.5,
            q_init="random",
        )
        self.tester.verify_1player("Grid", rl_config, 70_000)

    def test_OX(self):
        rl_config = ql.Config(
            epsilon=0.5,
            lr=0.1,
        )
        self.tester.verify_1player("OX", rl_config, 100_000)

    def test_Tiger(self):
        rl_config = ql.Config()
        rl_config.window_length = 10
        self.tester.verify_1player("Tiger", rl_config, 500_000)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_Grid", verbosity=2)
