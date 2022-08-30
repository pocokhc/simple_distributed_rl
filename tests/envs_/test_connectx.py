import unittest

import numpy as np
from envs import connectx  # noqa F401
from srl.base.define import EnvObservationType
from srl.base.env.spaces.box import BoxSpace
from srl.test import TestEnv
from srl.test.processor import TestProcessor


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestEnv()

    def test_play(self):
        env = self.tester.play_test("ConnectX")

        for x in [0, 2, 3, 4, 5, 6]:
            env.reset()
            board = [0] * 42
            self.assertTrue(not env.done)
            self.assertTrue((env.state == board).all())

            env.step(x)
            board[x + (5 * 7)] = 1
            self.assertTrue(not env.done)
            self.assertTrue((env.step_rewards == [0, 0]).all())
            self.assertTrue((env.state == board).all())

            env.step(1)
            board[1 + (5 * 7)] = 2
            self.assertTrue(not env.done)
            self.assertTrue((env.step_rewards == [0, 0]).all())
            self.assertTrue((env.state == board).all())

            env.step(x)
            board[x + (4 * 7)] = 1
            self.assertTrue(not env.done)
            self.assertTrue((env.step_rewards == [0, 0]).all())
            self.assertTrue((env.state == board).all())

            env.step(1)
            board[1 + (4 * 7)] = 2
            self.assertTrue(not env.done)
            self.assertTrue((env.step_rewards == [0, 0]).all())
            self.assertTrue((env.state == board).all())

            env.step(x)
            board[x + (3 * 7)] = 1
            self.assertTrue(not env.done)
            self.assertTrue((env.step_rewards == [0, 0]).all())
            self.assertTrue((env.state == board).all())

            env.step(1)
            board[1 + (3 * 7)] = 2
            self.assertTrue(not env.done)
            self.assertTrue((env.step_rewards == [0, 0]).all())
            self.assertTrue((env.state == board).all())

            env.step(x)
            board[x + (2 * 7)] = 1
            self.assertTrue(env.done)
            self.assertTrue((env.step_rewards == [1, -1]).all())
            self.assertTrue((env.state == board).all())

    def test_player(self):
        for player in [
            "alphabeta6",
            "alphabeta7",
            # "alphabeta8",
            # "alphabeta9",
            # "alphabeta10",
        ]:
            with self.subTest((player,)):
                self.tester.player_test("ConnectX", player)

    def test_processor(self):
        tester = TestProcessor()
        processor = connectx.LayerProcessor()
        env_name = "ConnectX"
        columns = 7
        rows = 6

        in_state = [0] * 42
        out_state = np.zeros((3, columns, rows))

        tester.run(processor, env_name)
        tester.change_observation_info(
            processor,
            env_name,
            EnvObservationType.SHAPE3,
            BoxSpace((3, columns, rows), 0, 1),
        )
        tester.observation_decode(
            processor,
            env_name,
            in_observation=in_state,
            out_observation=out_state,
        )


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_play", verbosity=2)
