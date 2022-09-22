import unittest

import numpy as np
from srl.utils.common import is_package_installed
from srl.utils.render_functions import print_to_text, text_to_rgb_array


class Test(unittest.TestCase):
    def test_print_to_text(self):
        for name, text in [
            ["str", "StubRender\nAAA"],
            ["none", ""],
            ["japanese", "あいうえお"],
        ]:
            with self.subTest((name,)):
                text2 = print_to_text(lambda: print(text))
                self.assertTrue(text2 == text + "\n")

    @unittest.skipUnless(is_package_installed("PIL"), "no module")
    def test_text_to_rgb_array(self):
        for name, text in [
            ["str", "StubRender\nAAA"],
            ["none", ""],
            ["japanese", "あいうえお"],
        ]:
            with self.subTest((name,)):
                rgb_array = text_to_rgb_array(text)
                self.assertTrue(len(rgb_array.shape) == 3)
                self.assertTrue(rgb_array.shape[2] == 3)
                self.assertTrue((rgb_array >= 0).all())
                self.assertTrue((rgb_array <= 255).all())
                self.assertTrue(rgb_array.dtype == np.uint8)

                # debug
                if False:
                    from PIL import Image

                    img = Image.fromarray(rgb_array)
                    img.show()


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_text_to_rgb_array", verbosity=2)
