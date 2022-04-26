import logging
from typing import Dict, Optional

from srl.base.env.base import EnvBase
from srl.base.env.gym_wrapper import GymWrapper
from srl.utils.common import load_module

logger = logging.getLogger(__name__)

_registry = {}


def make(id: str, kwargs: Optional[Dict] = None) -> EnvBase:
    if kwargs is None:
        kwargs = {}

    if id in _registry:
        env_cls = load_module(_registry[id]["entry_point"])

        _kwargs = _registry[id]["kwargs"].copy()
        _kwargs.update(kwargs)
        env = env_cls(**_kwargs)

        return env
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
