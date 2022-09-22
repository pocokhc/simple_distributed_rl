import unittest

import srl
from envs import grid  # noqa E401
from srl.runner import sequence
from srl.utils.common import is_package_installed, is_packages_installed


@unittest.skipUnless(is_packages_installed(["cv2", "matplotlib", "PIL", "pygame"]), "no module")
class Test(unittest.TestCase):
    def test_play(self):

        config = sequence.Config(srl.EnvConfig("Grid"), None)
        render = sequence.animation(config, max_steps=10)
        render.create_anime(draw_info=True).save("tmp/a.gif")

        config = sequence.Config(srl.EnvConfig("Grid"), None)
        render = sequence.animation(config, max_steps=10)
        render.create_anime(draw_info=True).save("tmp/b.gif")

    @unittest.skipUnless(is_package_installed("gym"), "no module")
    def test_gym(self):

        config = sequence.Config(srl.EnvConfig("MountainCar-v0"), None)
        render = sequence.animation(config, max_steps=10)
        render.create_anime(draw_info=True).save("tmp/c.gif")

        config = sequence.Config(srl.EnvConfig("MountainCar-v0"), None)
        render = sequence.animation(config, max_steps=10)
        render.create_anime(draw_info=True).save("tmp/d.gif")


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_play", verbosity=2)
