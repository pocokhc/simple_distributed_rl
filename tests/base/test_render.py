import unittest
from typing import Optional

import numpy as np
from srl.base.render import IRender, Render
from srl.utils.common import is_packages_installed


class StubConfig:
    def __init__(self):
        self.font_name = ""
        self.font_size = 12


class StubRender(IRender):
    def render_terminal(self, text, **kwargs) -> None:
        print(text)

    def render_rgb_array(self, return_rgb, **kwargs) -> Optional[np.ndarray]:
        if return_rgb:
            return np.zeros((4, 4, 3))
        else:
            return None


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.render = Render(StubRender(), StubConfig())

    def test_render_terminal(self):
        text = "StubRender\nAAA"
        with self.subTest(("return_text=False",)):
            self.render.render_terminal(return_text=False, text=text)

        with self.subTest(("return_text=True",)):
            text2 = self.render.render_terminal(return_text=True, text=text)
            self.assertTrue(text2 == text + "\n")

    @unittest.skipUnless(is_packages_installed(["cv2", "matplotlib", "PIL", "pygame"]), "no module")
    def test_render_rgb_array(self):
        text = "StubRender\nAAA"

        for return_rgb in [
            False,
            True,
        ]:
            with self.subTest((return_rgb,)):
                rgb_array = self.render.render_rgb_array(return_rgb=return_rgb, text=text)
                self.assertTrue(len(rgb_array.shape) == 3)
                self.assertTrue(rgb_array.shape[2] == 3)
                self.assertTrue((rgb_array >= 0).all())
                self.assertTrue((rgb_array <= 255).all())
                self.assertTrue(rgb_array.dtype == np.uint8)

    @unittest.skipUnless(is_packages_installed(["cv2", "matplotlib", "PIL", "pygame"]), "no module")
    def test_render_window(self):
        for _ in range(10):
            self.render.render_window(return_rgb=True)


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_render_rgb_array", verbosity=2)
