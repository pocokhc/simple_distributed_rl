import os
import unittest

from srl.rl import dummy
from srl.test import TestRL


class Test(TestRL, unittest.TestCase):
    def init_simple_check(self) -> None:
        print(os.environ)
        self.rl_config = dummy.Config()

    def test_simple_check_mp(self):
        if os.environ.get("SRL_MP_SKIP", "") == "1":
            return
        self.init_simple_check()
        self.simple_check(self.rl_config, is_mp=True, **self.simple_check_kwargs)
