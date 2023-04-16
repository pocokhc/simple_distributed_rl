import logging
import os
from typing import Optional

from srl.base.env.base import EnvRun
from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory, RLTrainer
from srl.base.rl.worker import WorkerRun
from srl.utils.common import load_module

logger = logging.getLogger(__name__)

_registry = {}
_registry_worker = {}

_ASSERT_MSG = "Run 'rl_config.reset_config(env)' first"


def make_remote_memory(
    rl_config: RLConfig,
    env: Optional[EnvRun] = None,
    return_class: bool = False,
    is_load: bool = True,
) -> RLRemoteMemory:
    if env is None:
        assert rl_config.is_set_env_config, _ASSERT_MSG
    else:
        rl_config.reset_config(env)
    name = rl_config.getName()
    _class = load_module(_registry[name][0])
    if return_class:
        return _class

    remote_memory = _class(rl_config)
    if is_load and rl_config.remote_memory_path != "":
        if not os.path.isfile(rl_config.remote_memory_path):
            logger.info(f"The file was not found and was not loaded.({rl_config.remote_memory_path})")
        else:
            remote_memory.load(rl_config.remote_memory_path)
    return remote_memory


def make_parameter(
    rl_config: RLConfig,
    env: Optional[EnvRun] = None,
    is_load: bool = True,
) -> RLParameter:
    if env is None:
        assert rl_config.is_set_env_config, _ASSERT_MSG
    else:
        rl_config.reset_config(env)
    name = rl_config.getName()
    parameter = load_module(_registry[name][1])(rl_config)
    if is_load and rl_config.parameter_path != "":
        if not os.path.isfile(rl_config.parameter_path):
            logger.info(f"The file was not found and was not loaded.({rl_config.parameter_path})")
        else:
            parameter.load(rl_config.parameter_path)
    return parameter


def make_trainer(
    rl_config: RLConfig,
    parameter: RLParameter,
    remote_memory: RLRemoteMemory,
    env: Optional[EnvRun] = None,
) -> RLTrainer:
    if env is None:
        assert rl_config.is_set_env_config, _ASSERT_MSG
    else:
        rl_config.reset_config(env)
    name = rl_config.getName()
    return load_module(_registry[name][2])(rl_config, parameter, remote_memory)


def make_worker(
    rl_config: RLConfig,
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
    env: Optional[EnvRun] = None,
    training: bool = False,
    distributed: bool = False,
    actor_id: int = 0,
) -> WorkerRun:
    if env is None:
        assert rl_config.is_set_env_config, _ASSERT_MSG
    else:
        rl_config.reset_config(env)
    name = rl_config.getName()
    worker = load_module(_registry[name][3])(
        rl_config,
        parameter,
        remote_memory,
        training,
        distributed,
        actor_id,
    )
    worker = WorkerRun(worker)

    # ExtendWorker
    if rl_config.extend_worker is not None:
        worker = rl_config.extend_worker(worker)
        worker = WorkerRun(worker)

    return worker


def make_worker_rulebase(name: str, **kwargs) -> WorkerRun:
    assert name in _registry_worker, f"{name} is not registered."
    worker = load_module(_registry_worker[name])(**kwargs)
    worker = WorkerRun(worker)
    return worker


def register(
    config: RLConfig,
    memory_entry_point: str,
    parameter_entry_point: str,
    trainer_entry_point: str,
    worker_entry_point: str,
):
    global _registry

    name = config.getName()
    assert name not in _registry, f"{name} was already registered."
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
    assert name not in _registry_worker, f"{name} was already registered."
    _registry_worker[name] = worker_entry_point
