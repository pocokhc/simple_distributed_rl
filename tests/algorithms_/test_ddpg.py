import unittest

from srl.test import TestRL
from srl.utils.common import is_package_installed

try:
    from algorithms import ddpg
except ModuleNotFoundError:
    pass


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_simple_check(self):
        self.tester.simple_check(ddpg.Config())

    def test_simple_check_mp(self):
        self.tester.simple_check_mp(ddpg.Config())

    def test_verify_Pendulum(self):
        rl_config = ddpg.Config()
        self.tester.verify_singleplay("Pendulum-v1", rl_config, 200 * 25)


if __name__ == "__main__":
    import __init__  # noqa F401

    unittest.main(module=__name__, defaultTest="Test.test_sequence", verbosity=2)
