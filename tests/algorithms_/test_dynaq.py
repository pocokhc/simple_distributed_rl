import unittest

from algorithms import dynaq
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_simple_check(self):
        self.tester.simple_check(dynaq.Config())

    def test_simple_check_mp(self):
        self.tester.simple_check_mp(dynaq.Config())

    def test_verify_grid(self):
        rl_config = dynaq.Config()
        self.tester.verify_singleplay("Grid", rl_config, 50_000)
        self.tester.verify_grid_policy()

    def test_verify_grid_mp(self):
        rl_config = dynaq.Config()
        self.tester.verify_singleplay("Grid", rl_config, 100_000, is_mp=True)
        self.tester.verify_grid_policy()


if __name__ == "__main__":
    import __init__  # noqa F401

    unittest.main(module=__name__, defaultTest="Test.test_verify_grid_mp", verbosity=2)
