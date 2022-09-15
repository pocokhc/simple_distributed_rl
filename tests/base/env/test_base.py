import unittest

import srl
from envs import grid
from srl.base.env.config import EnvConfig


class Test(unittest.TestCase):
    def test_config_copy(self):
        config = EnvConfig("Grid", {"move_prob": 1.0}, gym_prediction_by_simulation=False)
        env = srl.make_env(config)

        config2 = config.copy()
        self.assertTrue(config2.name == "Grid")
        self.assertDictEqual(config2.kwargs, {"move_prob": 1.0})
        self.assertTrue(not config2.gym_prediction_by_simulation)
        self.assertTrue(config2.max_episode_steps == env.max_episode_steps)
        self.assertTrue(config2.player_num == env.player_num)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_config_copy", verbosity=2)
