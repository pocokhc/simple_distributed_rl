import unittest

from srl.rl import dummy
from srl.test import TestRL


class Test(TestRL):
    def setUp(self) -> None:
        self.rl_config = dummy.Config()


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_simple_check", verbosity=2)
