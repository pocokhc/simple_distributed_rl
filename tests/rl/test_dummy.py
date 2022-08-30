import unittest

from srl.rl import dummy
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_simple_check(self):
        self.tester.simple_check(dummy.Config())

    def test_simple_check_mp(self):
        self.tester.simple_check_mp(dummy.Config())


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_simple_check", verbosity=2)
