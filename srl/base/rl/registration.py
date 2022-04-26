import logging
from typing import Optional, Tuple

from srl.base.env.env_for_rl import EnvForRL
from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory, RLTrainer, RLWorker
from srl.utils.common import load_module

logger = logging.getLogger(__name__)

_registry = {}


def make(config: RLConfig, env: Optional[EnvForRL]) -> Tuple[RLRemoteMemory, RLParameter, RLTrainer, RLWorker]:
    remote_memory = make_remote_memory(config, env)
    parameter = make_parameter(config, env)
    trainer = make_trainer(config, env, parameter, remote_memory)
    worker = make_worker(config, env, parameter, remote_memory)
    return remote_memory, parameter, trainer, worker


def make_remote_memory(config: RLConfig, env: Optional[EnvForRL], get_class: bool = False) -> RLRemoteMemory:
    if env is None:
        assert config.is_set_config_by_env, "Run set_config_by_env() first"
    else:
        config.set_config_by_env(env)
    name = config.getName()
    _class = load_module(_registry[name][0])
    if get_class:
        return _class
    else:
        return _class(config)


def make_parameter(config: RLConfig, env: Optional[EnvForRL]) -> RLParameter:
    if env is None:
        assert config.is_set_config_by_env, "Run set_config_by_env() first"
    else:
        config.set_config_by_env(env)
    name = config.getName()
    return load_module(_registry[name][1])(config)


def make_trainer(
    config: RLConfig, env: Optional[EnvForRL], parameter: RLParameter, remote_memory: RLRemoteMemory
) -> RLTrainer:
    if env is None:
        assert config.is_set_config_by_env, "Run set_config_by_env() first"
    else:
        config.set_config_by_env(env)
    name = config.getName()
    return load_module(_registry[name][2])(config, parameter, remote_memory)


def make_worker(
    config: RLConfig,
    env: Optional[EnvForRL],
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
    worker_id: int = 0,
) -> RLWorker:
    if env is None:
        assert config.is_set_config_by_env, "Run set_config_by_env() first"
    else:
        config.set_config_by_env(env)
    name = config.getName()
    return load_module(_registry[name][3])(config, parameter, remote_memory, worker_id)


def register(
    config_cls: RLConfig,
    memory_entry_point: str,
    parameter_entry_point: str,
    trainer_entry_point: str,
    worker_entry_point: str,
):
    global _registry

    name = config_cls.getName()
    if name in _registry:
        logger.warn(f"{name} was already registered. It will be overwritten.")
    _registry[name] = [
        memory_entry_point,
        parameter_entry_point,
        trainer_entry_point,
        worker_entry_point,
    ]
