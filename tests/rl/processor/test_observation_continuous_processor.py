import unittest

import gym.spaces
from srl.rl.processor.observation_continuous_processor import \
    ObservationContinuousProcessor


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = ObservationContinuousProcessor()


if __name__ == "__main__":
    name = "test_"
    unittest.main(module=__name__, defaultTest="Test." + name, verbosity=2)
