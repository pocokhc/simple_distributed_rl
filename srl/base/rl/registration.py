import logging
import os
from typing import Optional, Type

from srl.base.env.env_run import EnvRun
from srl.base.rl.base import (
    DummyRLMemory,
    DummyRLParameter,
    DummyRLTrainer,
    DummyRLWorker,
    IRLMemoryTrainer,
    IRLMemoryWorker,
    RLConfig,
    RLMemory,
    RLParameter,
    RLTrainer,
    RLWorker,
)
from srl.base.rl.config import DummyRLConfig
from srl.base.rl.worker_run import WorkerRun
from srl.utils.common import load_module

logger = logging.getLogger(__name__)

_registry = {}
_registry_worker = {}


def _check_rl_config(rl_config: RLConfig, env: Optional[EnvRun]) -> None:
    name = rl_config.getName()
    assert name in _registry, f"{name} is not registered."
    if env is None:
        assert rl_config.is_setup, "Run 'rl_config.reset(env)' first"
    else:
        rl_config.setup(env)
    rl_config.assert_params()


def make_memory(rl_config: RLConfig, env: Optional[EnvRun] = None, is_load: bool = True) -> RLMemory:
    _check_rl_config(rl_config, env)

    entry_point = _registry[rl_config.getName()][0]
    if entry_point == "dummy":
        memory: RLMemory = DummyRLMemory(rl_config)
    else:
        memory: RLMemory = load_module(entry_point)(rl_config)
    if is_load and rl_config.memory_path != "":
        if not os.path.isfile(rl_config.memory_path):
            logger.info(f"The file was not found and was not loaded.({rl_config.memory_path})")
        else:
            memory.load(rl_config.memory_path)
    return memory


def make_memory_class(rl_config: RLConfig, env: Optional[EnvRun] = None) -> Type[RLMemory]:
    _check_rl_config(rl_config, env)
    entry_point = _registry[rl_config.getName()][0]
    if entry_point == "dummy":
        memory_cls = DummyRLMemory
    else:
        memory_cls = load_module(entry_point)
    return memory_cls


def make_parameter(rl_config: RLConfig, env: Optional[EnvRun] = None, is_load: bool = True) -> RLParameter:
    _check_rl_config(rl_config, env)

    entry_point = _registry[rl_config.getName()][1]
    if entry_point == "dummy":
        parameter: RLParameter = DummyRLParameter(rl_config)
    else:
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
    memory: IRLMemoryTrainer,
    distributed: bool = False,
    train_only: bool = False,
    env: Optional[EnvRun] = None,
) -> RLTrainer:
    _check_rl_config(rl_config, env)
    entry_point = _registry[rl_config.getName()][2]
    if entry_point == "dummy":
        return DummyRLTrainer(rl_config, parameter, memory, distributed, train_only)
    else:
        return load_module(entry_point)(rl_config, parameter, memory, distributed, train_only)


def make_worker(
    rl_config: RLConfig,
    env: EnvRun,
    parameter: Optional[RLParameter] = None,
    memory: Optional[IRLMemoryWorker] = None,
    distributed: bool = False,
    actor_id: int = 0,
) -> WorkerRun:
    rl_config.setup(env, enable_log=True)
    _check_rl_config(rl_config, env)

    entry_point = _registry[rl_config.getName()][3]
    if entry_point == "dummy":
        worker: RLWorker = DummyRLWorker(rl_config, parameter, memory)
    else:
        worker: RLWorker = load_module(entry_point)(rl_config, parameter, memory)

    # ExtendWorker
    if rl_config.extend_worker is not None:
        worker = rl_config.extend_worker(worker, rl_config, parameter, memory)

    return WorkerRun(worker, env, distributed, actor_id)


def make_worker_rulebase(
    name: str,
    env: EnvRun,
    worker_kwargs: dict = {},
    distributed: bool = False,
    actor_id: int = 0,
) -> WorkerRun:
    # --- srl内はloadする
    if name not in _registry_worker:
        if name == "human":
            import srl.rl.human  # noqa F401
        elif name == "random":
            import srl.rl.random_play  # noqa F401

    # dummy config
    rl_config = DummyRLConfig(name=name)
    rl_config.setup(env, enable_log=False)

    assert name in _registry_worker, f"{name} is not registered."
    worker = load_module(_registry_worker[name])(config=rl_config, **worker_kwargs)
    return WorkerRun(worker, env, distributed, actor_id)


def register(
    config: RLConfig,
    memory_entry_point: str,
    parameter_entry_point: str,
    trainer_entry_point: str,
    worker_entry_point: str,
    enable_assert: bool = True,
) -> None:
    global _registry

    name = config.getName()
    if enable_assert:
        assert name not in _registry, f"{name} was already registered."
    else:
        if name in _registry:
            logger.warn(f"{name} was already registered. Not registered. entry_point={worker_entry_point}")
            return

    _registry[name] = [
        memory_entry_point,
        parameter_entry_point,
        trainer_entry_point,
        worker_entry_point,
    ]


def register_rulebase(
    name: str,
    entry_point: str,
    enable_assert: bool = True,
) -> None:
    global _registry_worker

    if enable_assert:
        assert name not in _registry, f"{name} was already registered."
    else:
        if name in _registry:
            logger.warn(f"{name} was already registered. Not registered. entry_point={entry_point}")
            return

    _registry_worker[name] = entry_point
