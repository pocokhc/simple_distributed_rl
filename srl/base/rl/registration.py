import logging
from typing import Optional, Tuple, Type

from srl.base.env.base import EnvBase
from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory, RLTrainer, RLWorker
from srl.utils.common import load_module

logger = logging.getLogger(__name__)

_registry = {}

_ASSERT_MSG = "Run 'rl_config.set_config_by_env(env)' first"


def make(config: RLConfig, env: EnvBase) -> Tuple[RLRemoteMemory, RLParameter, RLTrainer, RLWorker]:
    if not config.is_set_config_by_env:
        config.set_config_by_env(env)

    remote_memory = make_remote_memory(config)
    parameter = make_parameter(config)
    trainer = make_trainer(config, parameter, remote_memory)
    worker = make_worker(config, env, parameter, remote_memory)
    return remote_memory, parameter, trainer, worker


def make_remote_memory(config: RLConfig, get_class: bool = False) -> RLRemoteMemory:
    assert config.is_set_config_by_env, _ASSERT_MSG
    name = config.getName()
    _class = load_module(_registry[name][0])
    if get_class:
        return _class
    else:
        return _class(config)


def make_parameter(config: RLConfig) -> RLParameter:
    assert config.is_set_config_by_env, _ASSERT_MSG
    name = config.getName()
    return load_module(_registry[name][1])(config)


def make_trainer(config: RLConfig, parameter: RLParameter, remote_memory: RLRemoteMemory) -> RLTrainer:
    assert config.is_set_config_by_env, _ASSERT_MSG
    name = config.getName()
    return load_module(_registry[name][2])(config, parameter, remote_memory)


def make_worker(
    config: RLConfig,
    env: EnvBase,
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
    worker_id: int = 0,
) -> RLWorker:
    config.set_config_by_env(env)
    name = config.getName()
    worker = load_module(_registry[name][3])(config, parameter, remote_memory, worker_id)
    return worker


def register(
    config_cls: Type[RLConfig],
    memory_entry_point: str,
    parameter_entry_point: str,
    trainer_entry_point: str,
    worker_entry_point: str,
):
    global _registry

    name = config_cls.getName()
    if name in _registry:
        logger.warning(f"{name} was already registered. It will be overwritten.")
    _registry[name] = [
        memory_entry_point,
        parameter_entry_point,
        trainer_entry_point,
        worker_entry_point,
    ]
