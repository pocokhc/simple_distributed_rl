import unittest

import srl.envs.grid  # noqa F401
import srl.envs.oneroad  # noqa F401
from srl.algorithms import search_dynaq
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_Grid(self):
        rl_config = search_dynaq.Config()
        rl_config.ext_lr = 0.01
        parameter = self.tester.verify_1player("Grid", rl_config, 10_000)
        self.tester.verify_grid_policy(rl_config, parameter)

    def test_Grid_mp(self):
        rl_config = search_dynaq.Config()
        rl_config.ext_lr = 0.01
        self.tester.verify_1player("Grid", rl_config, 10_000, is_mp=True)

    def test_OneRoad(self):
        rl_config = search_dynaq.Config()
        self.tester.verify_1player("OneRoad", rl_config, 1_000)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_grid", verbosity=2)
