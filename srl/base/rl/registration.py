import logging
import os
from typing import List, Optional, Tuple, Type, Union, cast

from srl.base.define import PlayerType
from srl.base.env.env_run import EnvRun
from srl.base.rl.config import DummyRLConfig, RLConfig
from srl.base.rl.memory import DummyRLMemory, IRLMemoryWorker, RLMemory
from srl.base.rl.parameter import DummyRLParameter, RLParameter
from srl.base.rl.trainer import DummyRLTrainer, RLTrainer
from srl.base.rl.worker import DummyRLWorker, RLWorker
from srl.base.rl.worker_run import WorkerRun
from srl.utils.common import load_module

logger = logging.getLogger(__name__)

_registry = {"": ["", "", "", ""]}


def _check_rl_config(rl_config: RLConfig, env: Optional[EnvRun]) -> None:
    key = _create_registry_key(rl_config)
    if not isinstance(rl_config, DummyRLConfig):
        assert key in _registry, f"{key} is not registered."
    if env is None:
        assert rl_config.is_setup, "Run 'rl_config.reset(env)' first"
    else:
        rl_config.setup(env)
    rl_config.assert_params()


def _create_registry_key(rl_config: RLConfig) -> str:
    framework = rl_config.get_framework()
    if framework == "":
        key = rl_config.get_name()
    else:
        key = f"{rl_config.get_name()}:{framework}"
    return key


def make_memory(rl_config: RLConfig, env: Optional[EnvRun] = None, is_load: bool = True) -> RLMemory:
    _check_rl_config(rl_config, env)

    entry_point = _registry[_create_registry_key(rl_config)][0]
    if entry_point == "":
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
    entry_point = _registry[_create_registry_key(rl_config)][0]
    if entry_point == "":
        memory_cls = DummyRLMemory
    else:
        memory_cls = load_module(entry_point)
    return memory_cls


def make_parameter(rl_config: RLConfig, env: Optional[EnvRun] = None, is_load: bool = True) -> RLParameter:
    _check_rl_config(rl_config, env)

    entry_point = _registry[_create_registry_key(rl_config)][1]
    if entry_point == "":
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
    memory: RLMemory,
    env: Optional[EnvRun] = None,
) -> RLTrainer:
    _check_rl_config(rl_config, env)

    entry_point = _registry[_create_registry_key(rl_config)][2]
    if entry_point == "":
        return DummyRLTrainer(rl_config, parameter, memory)
    else:
        return load_module(entry_point)(rl_config, parameter, memory)


def make_worker(
    name_or_config: Union[str, RLConfig],
    env: EnvRun,
    parameter: Optional[RLParameter] = None,
    memory: Optional[IRLMemoryWorker] = None,
) -> WorkerRun:

    if isinstance(name_or_config, RLConfig):
        rl_config: RLConfig = name_or_config
        enable_log = True
    elif name_or_config == "human":
        import srl.rl.human

        rl_config: RLConfig = srl.rl.human.Config()
        enable_log = False
    elif name_or_config == "random":
        import srl.rl.random_play

        rl_config: RLConfig = srl.rl.random_play.Config()
        enable_log = False
    else:
        rl_config: RLConfig = DummyRLConfig(name=name_or_config) if isinstance(name_or_config, str) else name_or_config
        enable_log = False

    rl_config.setup(env, enable_log=enable_log)
    _check_rl_config(rl_config, env)

    if name_or_config == "dummy":
        worker: RLWorker = DummyRLWorker(rl_config, parameter, memory)
    else:
        entry_point = _registry[_create_registry_key(rl_config)][3]
        if entry_point == "":
            worker: RLWorker = DummyRLWorker(rl_config, parameter, memory)
        else:
            worker: RLWorker = load_module(entry_point)(rl_config, parameter, memory)

    # ExtendWorker
    if rl_config.extend_worker is not None:
        worker = rl_config.extend_worker(worker, rl_config, parameter, memory)

    return WorkerRun(worker, env)


def make_env_worker(
    env: EnvRun,
    name: str,
    worker_kwargs: dict = {},
    enable_raise: bool = True,
) -> Optional[WorkerRun]:
    worker = env.env.make_worker(name, **worker_kwargs)
    if worker is None:
        if enable_raise:
            raise ValueError(f"'{name}' worker is not found.")
        return None

    return WorkerRun(worker, env)


def make_workers(
    players: List[PlayerType],
    env: EnvRun,
    rl_config: Optional[RLConfig] = None,
    parameter: Optional[RLParameter] = None,
    memory: Optional[IRLMemoryWorker] = None,
    main_worker: Optional[WorkerRun] = None,
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
    for i, worker_type in enumerate(players):
        # --- none はベース、複数ある場合はparameterとmemoryのみ共有
        if worker_type is None:
            if (i == main_worker_idx) and (main_worker is not None):
                workers.append(main_worker)
            else:
                assert rl_config is not None
                w = rl_config.make_worker(env, parameter, memory)
                workers.append(w)
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
            w = env.make_worker(worker_type, worker_kwargs, enable_raise=False)
            if w is not None:
                workers.append(w)
                continue
            w = make_worker(worker_type, env)
            assert w is not None, f"not registered: {worker_type}"
            workers.append(w)
            continue

        # --- RLConfigは専用のWorkerを作成
        if isinstance(worker_type, object) and issubclass(worker_type.__class__, RLConfig):
            _rl_config = cast(RLConfig, worker_type)
            _rl_config.setup(env)
            _parameter = make_parameter(_rl_config)
            if worker_kwargs is not None:
                _parameter.restore(worker_kwargs)
            w = make_worker(
                _rl_config,
                env,
                _parameter,
                make_memory(_rl_config),
            )
            workers.append(w)
            continue

        raise ValueError(f"unknown worker: {worker_type}")

    return workers, main_worker_idx


def register(
    rl_config: RLConfig,
    memory_entry_point: str,
    parameter_entry_point: str,
    trainer_entry_point: str,
    worker_entry_point: str,
    check_duplicate: bool = True,
) -> None:
    global _registry

    key = _create_registry_key(rl_config)

    if check_duplicate:
        assert key not in _registry, f"{key} was already registered."
    elif key in _registry:
        # 既にあれば上書き
        logger.warning(f"{key} was already registered, but I overwrote it. entry_point={worker_entry_point}")

    _registry[key] = [
        memory_entry_point,
        parameter_entry_point,
        trainer_entry_point,
        worker_entry_point,
    ]


def register_rulebase(
    name_or_config: Union[str, RLConfig],
    entry_point: str,
    check_duplicate: bool = True,
) -> None:
    config: RLConfig = DummyRLConfig(name=name_or_config) if isinstance(name_or_config, str) else name_or_config
    register(config, "", "", "", entry_point, check_duplicate)
