from dataclasses import dataclass, fields

import pytest

from srl.base.env.config import EnvConfig
from srl.base.env.gym_user_wrapper import GymUserWrapper
from srl.base.env.processor import EnvProcessor
from tests.utils import assert_equal


@dataclass
class _DymmyProcessor(EnvProcessor):
    a: int = 10


def _gym_make_func(id, **kwargs):
    return None


class _GymWrapper(GymUserWrapper):
    pass


def _create_dummy_config(use_gym: bool):
    cfg = EnvConfig(
        "Dummy",
        kwargs={"a": 100, "b": ["a", 1]},
        max_episode_steps=2,
        episode_timeout=3,
        frameskip=4,
        random_noop_max=5,
        display_name="Dummy2",
        # gym(gymのテストはwrapperの方で)
        gym_make_func=None,
        gym_wrapper=None,
        use_gym=False,
        # processor
        processors=[_DymmyProcessor(11)],
    )
    if use_gym:
        pytest.importorskip("gymnasium")
        cfg.use_gym = True
        cfg.gym_make_func = _gym_make_func
        cfg.gym_wrapper = _GymWrapper()

    return cfg


@pytest.mark.parametrize("use_gym", [False, True])
def test_copy(use_gym):
    config = _create_dummy_config(use_gym)
    config2 = config.copy()
    assert config2.name == "Dummy2"

    for f in fields(config):
        print(f.name)
        assert_equal(getattr(config, f.name), getattr(config2, f.name))
    assert config.processors[0].a == config2.processors[0].a  # type: ignore


@pytest.mark.parametrize("file", ["json", "yaml"])
@pytest.mark.parametrize("use_gym", [False, True])
def test_save_load(tmpdir, file, use_gym):
    if file == "json":
        path = str(tmpdir / "config.json")
    elif file == "yaml":
        pytest.importorskip("yaml")
        path = str(tmpdir / "config.yaml")

    config = _create_dummy_config(use_gym)
    config.save(path)
    config2 = EnvConfig.load(path)

    for f in fields(config):
        print(f.name)
        assert_equal(getattr(config, f.name), getattr(config2, f.name))
    assert config.processors[0].a == config2.processors[0].a  # type: ignore


@pytest.mark.parametrize("use_gym", [False, True])
def test_summary(use_gym):
    config = _create_dummy_config(use_gym)
    config.summary()
