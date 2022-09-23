import unittest

import numpy as np
from srl.base.define import EnvObservationType
from srl.base.env.spaces.box import BoxSpace  # noqa F401
from srl.envs import grid
from srl.test import TestEnv
from srl.test.processor import TestProcessor


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestEnv()

    def test_grid(self):
        self.tester.play_test("Grid")

    def test_easy_grid(self):
        self.tester.play_test("EasyGrid")

    def test_processor(self):
        tester = TestProcessor()
        processor = grid.LayerProcessor()
        env_name = "Grid"

        env = grid.Grid()
        field = np.zeros((1, env.H, env.W))
        field[0][3][1] = 1

        tester.run(processor, env_name)
        tester.change_observation_info(
            processor,
            env_name,
            after_type=EnvObservationType.SHAPE3,
            after_space=BoxSpace((1, env.H, env.W), 0, 1),
        )
        tester.observation_decode(
            processor,
            env_name,
            in_observation=[1, 3],
            out_observation=field,
        )


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_processor", verbosity=2)
