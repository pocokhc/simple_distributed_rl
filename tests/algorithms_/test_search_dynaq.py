import unittest

from algorithms import search_dynaq
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_simple_check(self):
        self.tester.simple_check(search_dynaq.Config())

    def test_simple_check_mp(self):
        self.tester.simple_check_mp(search_dynaq.Config())

    def test_verify_grid(self):
        # TODO
        rl_config = search_dynaq.Config()
        rl_config.epsilon = 0.5
        rl_config.lr = 0.01
        self.tester.verify_singleplay("Grid", rl_config, 50_000)
        self.tester.verify_grid_policy()


if __name__ == "__main__":
    import __init__  # noqa F401

    unittest.main(module=__name__, defaultTest="Test.test_sequence", verbosity=2)
