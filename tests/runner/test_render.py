import unittest

import srl
from srl.runner import sequence
from srl.runner.callbacks import Rendering
from srl.utils import common as C


class Test(unittest.TestCase):
    def test_play(self):

        config = sequence.Config(
            env_config=srl.envs.Config("Grid"),
            rl_config=srl.rl.random_play.Config(),
        )

        render = Rendering(mode="", enable_animation=True)
        config.set_play_config(max_episodes=1, callbacks=[render])
        sequence.play(config)
        render.create_anime().save("tmp/a.gif")
        render.display()

        config = sequence.Config(
            env_config=srl.envs.Config("Grid"),
            rl_config=srl.rl.random_play.Config(),
        )
        render = Rendering(mode="", enable_animation=True)
        config.set_play_config(max_episodes=1, callbacks=[render])
        sequence.play(config)
        render.create_anime().save("tmp/b.gif")
        render.display()

    def test_gym(self):

        config = sequence.Config(
            env_config=srl.envs.Config("MountainCar-v0"),
            rl_config=srl.rl.random_play.Config(),
        )

        render = Rendering(mode="", enable_animation=True)
        config.set_play_config(max_episodes=1, callbacks=[render])
        sequence.play(config)
        render.create_anime().save("tmp/a.gif")
        render.display()

        config = sequence.Config(
            env_config=srl.envs.Config("MountainCar-v0"),
            rl_config=srl.rl.random_play.Config(),
        )
        render = Rendering(mode="", enable_animation=True)
        config.set_play_config(max_episodes=1, callbacks=[render])
        sequence.play(config)
        render.create_anime().save("tmp/b.gif")
        render.display()


if __name__ == "__main__":
    C.set_logger()
    unittest.main(module=__name__, defaultTest="Test.test_play", verbosity=2)
