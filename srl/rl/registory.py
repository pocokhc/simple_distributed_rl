import importlib
import logging

logger = logging.getLogger(__name__)

_rl_registry = {}


def make(name: str):
    module = importlib.import_module(_rl_registry[name])
    return module


def register(config, entry_point: str):
    global _rl_registry
    name = config.getName()
    if name in _rl_registry:
        logger.warn(f"{name} was already registered. It will be overwritten.")
    _rl_registry[name] = entry_point
