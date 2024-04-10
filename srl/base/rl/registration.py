import logging
import os
from typing import List, Optional, Tuple, Type, Union, cast

from srl.base.define import PlayerType
from srl.base.env.env_run import EnvRun
from srl.base.rl.config import DummyRLConfig, RLConfig
from srl.base.rl.memory import DummyRLMemory, IRLMemoryTrainer, IRLMemoryWorker, RLMemory
from srl.base.rl.parameter import DummyRLParameter, RLParameter
from srl.base.rl.trainer import DummyRLTrainer, RLTrainer
from srl.base.rl.worker import DummyRLWorker, RLWorker
from srl.base.rl.worker_run import WorkerRun
from srl.utils.common import load_module

logger = logging.getLogger(__name__)

_registry = {}
_registry_worker = {}


def _check_rl_config(rl_config: RLConfig, env: Optional[EnvRun]) -> None:
    name = rl_config.get_name()
    if not isinstance(rl_config, DummyRLConfig):
        assert name in _registry, f"{name} is not registered."
    if env is None:
        assert rl_config.is_setup, "Run 'rl_config.reset(env)' first"
    else:
        rl_config.setup(env)
    rl_config.assert_params()


def make_memory(rl_config: RLConfig, env: Optional[EnvRun] = None, is_load: bool = True) -> RLMemory:
    _check_rl_config(rl_config, env)

    if isinstance(rl_config, DummyRLConfig):
        memory: RLMemory = DummyRLMemory(rl_config)
    else:
        entry_point = _registry[rl_config.get_name()][0]
        memory: RLMemory = load_module(entry_point)(rl_config)
    if is_load and rl_config.memory_path != "":
        if not os.path.isfile(rl_config.memory_path):
            logger.info(f"The file was not found and was not loaded.({rl_config.memory_path})")
        else:
            memory.load(rl_config.memory_path)
    return memory


def make_memory_class(rl_config: RLConfig, env: Optional[EnvRun] = None) -> Type[RLMemory]:
    _check_rl_config(rl_config, env)
    if isinstance(rl_config, DummyRLConfig):
        memory_cls = DummyRLMemory
    else:
        entry_point = _registry[rl_config.get_name()][0]
        memory_cls = load_module(entry_point)
    return memory_cls


def make_parameter(rl_config: RLConfig, env: Optional[EnvRun] = None, is_load: bool = True) -> RLParameter:
    _check_rl_config(rl_config, env)

    if isinstance(rl_config, DummyRLConfig):
        parameter: RLParameter = DummyRLParameter(rl_config)
    else:
        entry_point = _registry[rl_config.get_name()][1]
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

    if isinstance(rl_config, DummyRLConfig):
        return DummyRLTrainer(rl_config, parameter, memory, distributed, train_only)
    else:
        entry_point = _registry[rl_config.get_name()][2]
        return load_module(entry_point)(rl_config, parameter, memory, distributed, train_only)


def make_worker(
    rl_config: RLConfig,
    env: EnvRun,
    parameter: Optional[RLParameter] = None,
    memory: Optional[IRLMemoryWorker] = None,
) -> WorkerRun:
    rl_config.setup(env, enable_log=True)
    _check_rl_config(rl_config, env)

    if isinstance(rl_config, DummyRLConfig):
        worker: RLWorker = DummyRLWorker(rl_config, parameter, memory)
    else:
        entry_point = _registry[rl_config.get_name()][3]
        worker: RLWorker = load_module(entry_point)(rl_config, parameter, memory)

    # ExtendWorker
    if rl_config.extend_worker is not None:
        worker = rl_config.extend_worker(worker, rl_config, parameter, memory)

    return WorkerRun(worker, env)


def make_worker_rulebase(name: str, env: EnvRun, worker_kwargs: dict = {}) -> WorkerRun:
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
    return WorkerRun(worker, env)


def make_workers(
    players: List[PlayerType],
    env: EnvRun,
    parameter: RLParameter,
    memory: RLMemory,
    rl_config: Optional[RLConfig] = None,
) -> Tuple[List[WorkerRun], int]:
    players = players[:]

    # 初期化されていない場合、一人目はNone、二人目以降はrandomにする
    for i in range(env.player_num):
        if i < len(players):
            continue
        if i == 0:
            players.append(None)
        else:
            players.append("random")

    # 最初に現れたNoneをmainにする
    main_worker_idx = 0
    for i, worker_type in enumerate(players):
        if worker_type is None:
            main_worker_idx = i
            break

    # --- make workers
    workers = []
    for worker_type in players:
        # --- none はベース
        if worker_type is None:
            assert rl_config is not None
            worker = make_worker(
                rl_config,
                env,
                parameter,
                memory,
            )
            workers.append(worker)
            continue

        if isinstance(worker_type, tuple) or isinstance(worker_type, list):
            worker_type, worker_kwargs = worker_type
        else:
            worker_kwargs = None

        # --- 文字列はenv側またはルールベースのアルゴリズム
        if isinstance(worker_type, str):
            if worker_kwargs is None:
                worker_kwargs = {}
            worker_kwargs = cast(dict, worker_kwargs)
            worker = env.make_worker(worker_type, enable_raise=False)
            if worker is not None:
                workers.append(worker)
                continue
            worker = make_worker_rulebase(worker_type, env, worker_kwargs=worker_kwargs)
            assert worker is not None, f"not registered: {worker_type}"
            workers.append(worker)
            continue

        # --- RLConfigは専用のWorkerを作成
        if isinstance(worker_type, object) and issubclass(worker_type.__class__, RLConfig):
            _rl_config = cast(RLConfig, worker_type)
            _rl_config.setup(env)
            _parameter = make_parameter(_rl_config)
            if worker_kwargs is not None:
                _parameter.restore(worker_kwargs)
            worker = make_worker(
                _rl_config,
                env,
                _parameter,
                make_memory(_rl_config),
            )
            workers.append(worker)
            continue

        raise ValueError(f"unknown worker: {worker_type}")

    return workers, main_worker_idx


def register(
    config: RLConfig,
    memory_entry_point: str,
    parameter_entry_point: str,
    trainer_entry_point: str,
    worker_entry_point: str,
    enable_assert: bool = True,
) -> None:
    global _registry

    name = config.get_name()
    if enable_assert:
        assert name not in _registry, f"{name} was already registered."
    elif name in _registry:
        # 既にあれば上書き
        logger.warning(f"{name} was already registered, but I overwrote it. entry_point={worker_entry_point}")

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
    elif name in _registry:
        # 既にあれば上書き
        logger.warning(f"{name} was already registered, but I overwrote it. entry_point={entry_point}")

    _registry_worker[name] = entry_point
