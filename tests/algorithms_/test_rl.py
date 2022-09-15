import glob
import importlib.machinery as imm
import os
import unittest

import algorithms
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()
        self.rl_list = []

        for path in glob.glob(os.path.join(list(algorithms.__path__)[0], "*.py")):
            if os.path.basename(path).startswith("_"):
                continue

            module = imm.SourceFileLoader(os.path.basename(path), path).load_module()

            if os.path.basename(path) in [
                "muzero.py",
                "stochastic_muzero.py",
            ]:
                enable_image = True
            else:
                enable_image = False
            self.rl_list.append([module.Config(), enable_image])

    def test_simple_check(self):
        tester = TestRL()
        for rl_config, enable_image in self.rl_list:
            with self.subTest(rl_config.getName()):
                tester.simple_check(
                    rl_config,
                    enable_image=enable_image,
                    check_render=False,
                )

    # py ファイルからloadしたモジュールはpickle化できないのでテスト不可
    # def test_simple_check_mp(self):
    #    tester = TestRL()
    #    for rl_config, enable_image in self.rl_list:
    #        with self.subTest(rl_config.getName()):
    #            tester.simple_check(rl_config, enable_image=enable_image, is_mp=True)
    #        break


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_simple_check_mp", verbosity=2)
