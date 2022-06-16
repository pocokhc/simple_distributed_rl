import unittest

from srl.utils import common as C


class Test(unittest.TestCase):
    def test_is_package_installed(self):
        self.assertTrue(C.is_package_installed("numpy"))
        self.assertTrue(C.is_package_installed("numpy"))
        self.assertTrue(not C.is_package_installed("aaaaaa"))
        self.assertTrue(not C.is_package_installed("aaaaaa"))

    def test_is_env_notebook(self):
        self.assertFalse(C.is_env_notebook())


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_is_env_notebook", verbosity=2)
