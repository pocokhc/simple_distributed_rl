import logging
from typing import Dict

from srl.base.env.base import EnvBase
from srl.base.env.env_for_rl import EnvConfig, EnvForRL
from srl.base.env.gym_wrapper import GymWrapper
from srl.base.rl.base import RLConfig
from srl.utils.common import load_module

logger = logging.getLogger(__name__)

_registry = {}


def make(
    env_config: EnvConfig,
    rl_config: RLConfig,
) -> EnvForRL:
    env = make_env(env_config.name, env_config.kwargs)
    env = EnvForRL(env, rl_config, env_config)
    return env


def make_env(
    id: str,
    env_kwargs: Dict = None,
) -> EnvBase:
    if env_kwargs is None:
        env_kwargs = {}

    if id in _registry:
        env_cls = load_module(_registry[id]["entry_point"])

        _kwargs = _registry[id]["kwargs"].copy()
        _kwargs.update(env_kwargs)
        env = env_cls(**_kwargs)

    else:
        # gym env
        env = GymWrapper(id)

    return env


def register(id: str, entry_point: str, kwargs: Dict = None) -> None:
    global _registry
    if kwargs is None:
        kwargs = {}

    if id in _registry:
        logger.warn(f"{id} was already registered. It will be overwritten.")
    _registry[id] = {
        "entry_point": entry_point,
        "kwargs": kwargs,
    }
