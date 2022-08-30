import unittest

import numpy as np
from envs import ox  # noqa F401
from srl.base.define import EnvObservationType
from srl.base.env.spaces.box import BoxSpace
from srl.test import TestEnv
from srl.test.processor import TestProcessor


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestEnv()

    def test_play(self):
        self.tester.play_test("OX")

    def test_player(self):
        self.tester.player_test("OX", "cpu")

    def test_processor(self):
        tester = TestProcessor()
        processor = ox.LayerProcessor()
        env_name = "OX"

        in_state = [0] * 9
        out_state = np.zeros((2, 3, 3))

        tester.run(processor, env_name)
        tester.change_observation_info(
            processor,
            env_name,
            EnvObservationType.SHAPE3,
            BoxSpace((2, 3, 3), 0, 1),
        )
        tester.observation_decode(
            processor,
            env_name,
            in_observation=in_state,
            out_observation=out_state,
        )


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_play", verbosity=2)
