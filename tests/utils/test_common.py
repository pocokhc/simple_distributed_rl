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

    def test_compare_less_version(self):
        self.assertTrue(C.compare_less_version("1.2.a3", "2.0.0"))
        self.assertFalse(C.compare_less_version("2.0.0", "1.2.a3"))
        self.assertFalse(C.compare_less_version("3.0.0", "3.0.0"))

    @unittest.skipUnless(C.is_package_installed("tensorflow"), "no module")
    def test_is_enable_device_name(self):
        self.assertTrue(C.is_enable_device_name("/CPU:0"))
        self.assertFalse(C.is_enable_device_name("/CPU:99999"))


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_is_enable_device_name", verbosity=2)
