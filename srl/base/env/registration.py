import logging
from typing import Dict, Union

from srl.base.env.base import EnvBase, EnvConfig
from srl.base.env.gym_wrapper import GymWrapper
from srl.utils.common import is_package_installed, load_module

logger = logging.getLogger(__name__)

_registry = {}


def make(config: Union[str, EnvConfig]) -> EnvBase:
    if isinstance(config, str):
        config = EnvConfig(config)

    env_name = config.name
    if env_name in _registry:
        env_cls = load_module(_registry[env_name]["entry_point"])

        _kwargs = _registry[env_name]["kwargs"].copy()
        _kwargs.update(config.kwargs)
        env = env_cls(**_kwargs)
        return env

    # --- gym
    if is_package_installed("gym"):
        return GymWrapper(env_name, config.gym_prediction_by_simulation)

    raise ValueError(f"'{env_name}' is not found.")


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
