import unittest

from srl.algorithms import search_dynaq
from srl.test import TestRL
import srl.envs.grid  # noqa F401
import srl.envs.oneroad  # noqa F401


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_simple_check(self):
        self.tester.simple_check(search_dynaq.Config())

    def test_simple_check_mp(self):
        self.tester.simple_check_mp(search_dynaq.Config())

    def test_verify_grid(self):
        rl_config = search_dynaq.Config()
        rl_config.ext_lr = 0.01
        self.tester.verify_singleplay("Grid", rl_config, 10_000)
        self.tester.verify_grid_policy()

    def test_verify_grid_mp(self):
        rl_config = search_dynaq.Config()
        rl_config.ext_lr = 0.01
        self.tester.verify_singleplay("Grid", rl_config, 10_000, is_mp=True)

    def test_verify_oneroad(self):
        rl_config = search_dynaq.Config()
        self.tester.verify_singleplay("OneRoad", rl_config, 1_000)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_grid", verbosity=2)
