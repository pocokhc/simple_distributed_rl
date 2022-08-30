import unittest

from algorithms import ql
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_simple_check(self):
        self.tester.simple_check(ql.Config())

    def test_simple_check_mp(self):
        self.tester.simple_check_mp(ql.Config())

    def test_verify_grid(self):
        rl_config = ql.Config(
            epsilon=0.5,
            lr=0.01,
        )
        self.tester.verify_singleplay("Grid", rl_config, 50_000)
        self.tester.verify_grid_policy()

    def test_verify_grid_mp(self):
        rl_config = ql.Config(
            epsilon=0.5,
            lr=0.01,
        )
        self.tester.verify_singleplay("Grid", rl_config, 50_000, is_mp=True)
        self.tester.verify_grid_policy()

    def test_verify_grid_random(self):
        rl_config = ql.Config(
            epsilon=0.5,
            q_init="random",
        )
        self.tester.verify_singleplay("Grid", rl_config, 70_000)

    def test_verify_ox(self):
        rl_config = ql.Config(
            epsilon=0.5,
            lr=0.1,
        )
        self.tester.verify_2play("OX", rl_config, 100_000)

    def test_verify_tiger(self):
        rl_config = ql.Config()
        rl_config.window_length = 10
        self.tester.verify_singleplay("Tiger", rl_config, 50_000)


if __name__ == "__main__":
    import __init__  # noqa F401

    unittest.main(module=__name__, defaultTest="Test.test_verify_tiger", verbosity=2)
