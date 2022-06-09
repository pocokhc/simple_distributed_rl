import unittest

import srl
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.rl_list = []
        for name in srl.rl.__dict__.keys():
            if name.startswith("_"):
                continue
            if name.startswith("make"):
                continue
            if name in [
                "memories",
                "functions",
                "human",
                "random_play",
            ]:
                continue
            self.rl_list.append(srl.rl.__dict__[name].Config())

    def test_sequence(self):
        tester = TestRL()
        for rl_config in self.rl_list:
            with self.subTest(rl_config.getName()):
                tester.play_sequence(rl_config)

    def test_mp(self):
        tester = TestRL()
        for rl_config in self.rl_list:
            with self.subTest(rl_config.getName()):
                tester.play_mp(rl_config)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_mp", verbosity=2)
