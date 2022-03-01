import logging

logger = logging.getLogger(__name__)

_memory_registry = {}


def make(config):
    return _memory_registry[config.getName()](config)


def get_class(config):
    return _memory_registry[config.getName()]


def register(config, memory_class):
    global _memory_registry
    name = config.getName()
    if name in _memory_registry:
        logger.warn(f"{name} was already registered. It will be overwritten.")
    _memory_registry[name] = memory_class
