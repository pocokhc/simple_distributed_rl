import unittest

from srl.test import TestRL
from srl.utils.common import is_package_installed

try:
    from srl.algorithms import sac
except ModuleNotFoundError:
    pass


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_simple_check(self):
        self.tester.simple_check(sac.Config())

    def test_simple_check_mp(self):
        self.tester.simple_check_mp(sac.Config())

    def test_verify_Pendulum(self):
        rl_config = sac.Config()
        self.tester.verify_singleplay("Pendulum-v1", rl_config, 200 * 25)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_sequence", verbosity=2)
