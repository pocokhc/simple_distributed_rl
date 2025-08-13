import glob
import os

import pytest

import srl
from srl.base.exception import UndefinedError


def pytest_generate_tests(metafunc):
    if {"env_path", "rl_path", "context_path"} <= set(metafunc.fixturenames):
        params = []
        for rl_fn in glob.glob(os.path.join("configs", "algorithms", "*.yaml")):
            for env_fn in glob.glob(os.path.join("configs", "envs", "*.yaml")):
                for cont_fn in glob.glob(os.path.join("configs", "context", "*.yaml")):
                    params.append((env_fn, rl_fn, cont_fn))
        metafunc.parametrize(("env_path", "rl_path", "context_path"), params)


def test_play(env_path, rl_path, context_path):
    pytest.importorskip("yaml")
    import yaml

    try:
        env_config = srl.load_env(yaml.safe_load(open(env_path)))
    except ModuleNotFoundError:
        pytest.skip(f"ModuleNotFoundError({env_path})")

    rl_config = srl.load_rl(yaml.safe_load(open(rl_path)))
    context = srl.load_context(yaml.safe_load(open(context_path)))
    context.timeout = 1

    try:
        runner = srl.Runner(env_config, rl_config, context)
    except UndefinedError:
        pytest.skip(f"env is not found.({env_path})")
    runner.summary(show_changed_only=True)
    runner.play()
