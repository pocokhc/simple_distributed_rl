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

    def test_Pendulum(self):
        rl_config = sac.Config()
        self.tester.verify_1player("Pendulum-v1", rl_config, 200 * 25)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_sequence", verbosity=2)
