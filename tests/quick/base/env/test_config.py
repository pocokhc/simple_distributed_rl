import pytest

from srl.base.env.config import EnvConfig
from srl.base.exception import UndefinedError


def test_copy():
    config = EnvConfig(
        "Dummy",
        kwargs={
            "a": 100,
            "b": ["a", 1],
        },
    )
    config2 = config.copy()
    assert config2.id == "Dummy"
    assert config2.kwargs["a"] == 100
    assert config2.kwargs["b"][0] == "a"
    assert config2.kwargs["b"][1] == 1

    with pytest.raises(UndefinedError):
        assert config2.name == "Dummy"


def test_summary():
    cfg = EnvConfig(
        "Dummy",
        kwargs={
            "a": 100,
            "b": ["a", 1],
        },
    )
    cfg.summary()
