import unittest

import srl
from srl.runner import sequence


class Test(unittest.TestCase):
    def test_play(self):

        config = sequence.Config(srl.EnvConfig("Grid"), None)
        render = sequence.animation(config, max_steps=10)
        render.create_anime(draw_info=True).save("tmp/a.gif")

        config = sequence.Config(srl.EnvConfig("Grid"), None)
        render = sequence.animation(config, max_steps=10)
        render.create_anime(draw_info=True).save("tmp/b.gif")

    def test_gym(self):

        config = sequence.Config(srl.EnvConfig("MountainCar-v0"), None)
        render = sequence.animation(config, max_steps=10)
        render.create_anime(draw_info=True).save("tmp/c.gif")

        config = sequence.Config(srl.EnvConfig("MountainCar-v0"), None)
        render = sequence.animation(config, max_steps=10)
        render.create_anime(draw_info=True).save("tmp/d.gif")


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_play", verbosity=2)
