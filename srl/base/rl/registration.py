import logging
import os
from typing import Optional, Type

from srl.base.env.env_run import EnvRun
from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory, RLTrainer
from srl.base.rl.config import DummyConfig
from srl.base.rl.worker import WorkerBase
from srl.base.rl.worker_run import WorkerRun
from srl.utils.common import load_module

logger = logging.getLogger(__name__)

_registry = {}
_registry_worker = {}


def _check_rl_config(
    rl_config: RLConfig,
    env: Optional[EnvRun] = None,
):
    name = rl_config.getName()
    assert name in _registry, f"{name} is not registered."

    if env is None:
        assert rl_config.is_reset, "Run 'rl_config.reset(env)' first"
    else:
        rl_config.reset(env)
    rl_config.assert_params()
    return rl_config


def make_remote_memory(
    rl_config: RLConfig,
    env: Optional[EnvRun] = None,
    return_class: bool = False,
    is_load: bool = True,
) -> RLRemoteMemory:
    rl_config = _check_rl_config(rl_config, env)

    entry_point = _registry[rl_config.getName()][0]
    _class: Type[RLRemoteMemory] = load_module(entry_point)
    if return_class:
        return _class  # type: ignore , Type missing OK

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
    rl_config = _check_rl_config(rl_config, env)

    entry_point = _registry[rl_config.getName()][1]
    parameter: RLParameter = load_module(entry_point)(rl_config)
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
    rl_config = _check_rl_config(rl_config, env)
    entry_point = _registry[rl_config.getName()][2]
    return load_module(entry_point)(rl_config, parameter, remote_memory)


def make_worker(
    rl_config: RLConfig,
    env: EnvRun,
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
    distributed: bool = False,
    actor_id: int = 0,
) -> WorkerRun:
    rl_config = _check_rl_config(rl_config, env)
    entry_point = _registry[rl_config.getName()][3]
    worker: WorkerBase = load_module(entry_point)(rl_config, parameter, remote_memory)

    # ExtendWorker
    if rl_config.extend_worker is not None:
        worker = rl_config.extend_worker(worker, rl_config, parameter, remote_memory)

    return WorkerRun(worker, env, distributed, actor_id)


def make_worker_rulebase(
    name: str,
    env: EnvRun,
    update_config_parameter: dict = {},
    distributed: bool = False,
    actor_id: int = 0,
    is_reset_logger: bool = True,
) -> WorkerRun:
    # --- srl内はloadする
    if name == "human":
        import srl.rl.human  # noqa F401
    elif name == "random":
        import srl.rl.random_play  # noqa F401

    rl_config = DummyConfig(name=name)
    name = rl_config.getName()

    # --- config update
    for k, v in update_config_parameter.items():
        setattr(rl_config, k, v)

    assert name in _registry_worker, f"{name} is not registered."
    worker = load_module(_registry_worker[name])(rl_config, None, None)
    return WorkerRun(worker, env, distributed, actor_id, is_reset_logger=is_reset_logger)


def register(
    config: RLConfig,
    memory_entry_point: str,
    parameter_entry_point: str,
    trainer_entry_point: str,
    worker_entry_point: str,
) -> None:
    global _registry

    name = config.getName()
    assert name not in _registry, f"{name} was already registered."
    _registry[name] = [
        memory_entry_point,
        parameter_entry_point,
        trainer_entry_point,
        worker_entry_point,
    ]


def register_rulebase(
    name: str,
    worker_entry_point: str,
) -> None:
    global _registry_worker
    assert name not in _registry_worker, f"{name} was already registered."
    _registry_worker[name] = worker_entry_point
