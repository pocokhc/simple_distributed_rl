import unittest

from srl.test import TestEnv


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestEnv()

    def test_play(self):
        self.tester.play_test("ConnectX")

    def test_player(self):
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
    unittest.main(module=__name__, defaultTest="Test.test_player", verbosity=2)
