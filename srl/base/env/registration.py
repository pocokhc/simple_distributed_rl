import logging
from typing import Dict, Union

from srl.base.env.base import EnvConfig, EnvRun
from srl.base.env.gym_wrapper import GymWrapper
from srl.utils.common import is_package_installed, load_module

logger = logging.getLogger(__name__)

_registry = {}


def make(config: Union[str, EnvConfig]) -> EnvRun:
    if isinstance(config, str):
        config = EnvConfig(config)

    env_name = config.name
    if env_name in _registry:
        env_cls = load_module(_registry[env_name]["entry_point"])

        _kwargs = _registry[env_name]["kwargs"].copy()
        _kwargs.update(config.kwargs)
        env = env_cls(**_kwargs)

    elif is_package_installed("gym"):
        env = GymWrapper(env_name, config.gym_prediction_by_simulation)
    else:
        raise ValueError(f"'{env_name}' is not found.")

    return EnvRun(env)


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
