import unittest

import srl
from srl.runner import sequence


class Test(unittest.TestCase):
    def test_play(self):

        config = sequence.Config(srl.envs.Config("Grid"), None)
        _, render = sequence.render(config, max_steps=10, enable_animation=True)
        render.create_anime().save("tmp/a.gif")

        config = sequence.Config(srl.envs.Config("Grid"), None)
        _, render = sequence.render(config, max_steps=10, enable_animation=True)
        render.create_anime().save("tmp/b.gif")

    def test_gym(self):

        config = sequence.Config(srl.envs.Config("MountainCar-v0"), None)
        _, render = sequence.render(config, max_steps=10, enable_animation=True)
        render.create_anime().save("tmp/c.gif")

        config = sequence.Config(srl.envs.Config("MountainCar-v0"), None)
        _, render = sequence.render(config, max_steps=10, enable_animation=True)
        render.create_anime().save("tmp/d.gif")


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_play", verbosity=2)
