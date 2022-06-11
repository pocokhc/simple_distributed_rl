import unittest

import srl
from srl.runner import sequence
from srl.utils import common as C


class Test(unittest.TestCase):
    def test_play(self):

        config = sequence.Config(srl.envs.Config("Grid"), None)
        _, render = sequence.render(config, max_steps=10, mode="", enable_animation=True)
        render.create_anime().save("tmp/a.gif")
        render.display()

        config = sequence.Config(srl.envs.Config("Grid"), None)
        _, render = sequence.render(config, max_steps=10, mode="", enable_animation=True)
        render.create_anime().save("tmp/b.gif")
        render.display()

    def test_gym(self):

        config = sequence.Config(srl.envs.Config("MountainCar-v0"), None)
        _, render = sequence.render(config, max_steps=10, mode="", enable_animation=True)
        render.create_anime().save("tmp/c.gif")
        render.display()

        config = sequence.Config(srl.envs.Config("MountainCar-v0"), None)
        _, render = sequence.render(config, max_steps=10, mode="", enable_animation=True)
        render.create_anime().save("tmp/d.gif")
        render.display()


if __name__ == "__main__":
    C.set_logger()
    unittest.main(module=__name__, defaultTest="Test.test_play", verbosity=2)
