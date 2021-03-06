import logging
import os
from typing import Optional, Tuple, Type

from srl.base.env.base import EnvRun
from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory, RLTrainer, WorkerRun
from srl.utils.common import load_module

logger = logging.getLogger(__name__)

_registry = {}
_registry_worker = {}

_ASSERT_MSG = "Run 'rl_config.set_config_by_env(env)' first"


def make(rl_config: RLConfig, env: EnvRun) -> Tuple[RLRemoteMemory, RLParameter, RLTrainer, WorkerRun]:
    if not rl_config.is_set_config_by_env:
        rl_config.set_config_by_env(env)

    remote_memory = make_remote_memory(rl_config)
    parameter = make_parameter(rl_config)
    trainer = make_trainer(rl_config, parameter, remote_memory)
    worker = make_worker(rl_config, env, parameter, remote_memory)
    return remote_memory, parameter, trainer, worker


def make_remote_memory(rl_config: RLConfig, get_class: bool = False) -> RLRemoteMemory:
    assert rl_config.is_set_config_by_env, _ASSERT_MSG
    name = rl_config.getName()
    _class = load_module(_registry[name][0])
    if get_class:
        return _class
    else:
        remote_memory = _class(rl_config)
        if rl_config.remote_memory_path != "":
            if not os.path.isfile(rl_config.remote_memory_path):
                logger.info(f"The file was not found and was not loaded.({rl_config.remote_memory_path})")
            else:
                remote_memory.load(rl_config.remote_memory_path)
        return remote_memory


def make_parameter(rl_config: RLConfig) -> RLParameter:
    assert rl_config.is_set_config_by_env, _ASSERT_MSG
    name = rl_config.getName()
    parameter = load_module(_registry[name][1])(rl_config)
    if rl_config.parameter_path != "":
        if not os.path.isfile(rl_config.parameter_path):
            logger.info(f"The file was not found and was not loaded.({rl_config.parameter_path})")
        else:
            parameter.load(rl_config.parameter_path)
    return parameter


def make_trainer(rl_config: RLConfig, parameter: RLParameter, remote_memory: RLRemoteMemory) -> RLTrainer:
    assert rl_config.is_set_config_by_env, _ASSERT_MSG
    name = rl_config.getName()
    return load_module(_registry[name][2])(rl_config, parameter, remote_memory)


def make_worker(
    rl_config: RLConfig,
    env: EnvRun,
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
    actor_id: int = 0,
) -> WorkerRun:
    if not rl_config.is_set_config_by_env:
        rl_config.set_config_by_env(env)

    name = rl_config.getName()
    worker = load_module(_registry[name][3])(rl_config, parameter, remote_memory, actor_id)
    worker = WorkerRun(worker)

    # ExtendWorker
    if rl_config.extend_worker is not None:
        worker = rl_config.extend_worker(worker, env)
        worker = WorkerRun(worker)

    worker.set_play_info(False, False)
    return worker


def make_worker_rulebase(name: str, **kwargs) -> Optional[WorkerRun]:
    if name not in _registry_worker:
        return None
    worker = load_module(_registry_worker[name])(**kwargs)
    worker = WorkerRun(worker)
    worker.set_play_info(False, False)
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


def register_worker(
    name: str,
    worker_entry_point: str,
):
    global _registry_worker

    if name in _registry_worker:
        logger.warning(f"{name} was already registered. It will be overwritten.")
    _registry_worker[name] = worker_entry_point
