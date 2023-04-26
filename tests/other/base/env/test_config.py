from srl.base.env.config import EnvConfig


def test_copy():
    config = EnvConfig("Dummy")

    config2 = config.copy()
    assert config2.name == "Dummy"
