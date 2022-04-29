import unittest

from srl import rl
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_play(self):
        rl_config = rl.c51.Config()
        self.tester.play_test(self, rl_config)

    def test_verify_Grid(self):
        rl_config = rl.c51.Config(
            epsilon=0.5,
            lr=0.002,
            hidden_layer_sizes=(16, 16),
            categorical_num_atoms=11,
            categorical_v_min=-2,
            categorical_v_max=2,
        )
        self.tester.play_verify_singleplay(self, "Grid", rl_config, 4000, 100)


if __name__ == "__main__":
    # unittest.main(module=__name__, defaultTest="Test.test_play", verbosity=2)
    unittest.main(module=__name__, defaultTest="Test.test_verify_Grid", verbosity=2)
