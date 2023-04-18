import unittest

from srl.base.env.config import EnvConfig


class Test(unittest.TestCase):
    def test_copy(self):
        config = EnvConfig("Dummy")

        config2 = config.copy()
        assert config2.name == "Dummy"
