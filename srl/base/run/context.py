import copy
import enum
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union, cast

from srl.base.define import RenderModes
from srl.base.env.config import EnvConfig
from srl.base.env.env_run import EnvRun
from srl.base.env.registration import make as make_env
from srl.base.rl.base import RLMemory, RLParameter
from srl.base.rl.config import RLConfig
from srl.base.rl.registration import make_memory, make_parameter, make_worker, make_worker_rulebase
from srl.base.rl.worker_run import WorkerRun
from srl.utils.serialize import convert_for_json

logger = logging.getLogger(__name__)


class RunNameTypes(enum.Enum):
    main = enum.auto()
    trainer = enum.auto()
    actor = enum.auto()
    eval = enum.auto()


StrWorkerType = Union[str, Tuple[str, dict]]
RLWorkerType = Union[RLConfig, Tuple[RLConfig, Any]]  # [RLConfig, RLParameter]


@dataclass
class RunContext:
    """
    実行時の状態ををまとめたクラス
    A class that summarizes the runtime state
    """

    env_config: EnvConfig
    rl_config: RLConfig

    # playersという変数名だけど、役割はworkersの方が正しい
    players: List[Union[None, StrWorkerType, RLWorkerType]] = field(default_factory=list)

    # --- play context
    run_name: RunNameTypes = RunNameTypes.main
    # stop config
    max_episodes: int = 0
    timeout: float = 0
    max_steps: int = 0
    max_train_count: int = 0
    max_memory: int = 0
    # play config
    shuffle_player: bool = True
    disable_trainer: bool = False
    # play info
    distributed: bool = False
    training: bool = False
    render_mode: Union[str, RenderModes] = RenderModes.none

    # --- mp
    actor_id: int = 0
    actor_num: int = 1

    # --- random
    seed: Optional[int] = None

    # --- device
    framework: str = ""
    enable_tf_device: bool = True
    used_device_tf: str = "/CPU"
    used_device_torch: str = "cpu"

    def create_controller(self) -> "RunContextController":
        return RunContextController(self)


class RunContextController:
    def __init__(self, context: RunContext):
        self.context = context

    def to_dict(self, skip_config: bool = False) -> dict:
        if skip_config:
            dat: dict = {}
            for k, v in self.context.__dict__.items():
                if k in ["env_config", "rl_config"]:
                    continue
                dat[k] = v
            dat: dict = convert_for_json(dat)
        else:
            dat: dict = convert_for_json(self.context.__dict__)
        return dat

    def copy(self) -> "RunContext":
        return copy.deepcopy(self.context)

    def set_device(self, framework, used_device_tf, used_device_torch):
        self.context.framework = framework
        self.context.used_device_tf = used_device_tf
        self.context.used_device_torch = used_device_torch
        self.context.rl_config._used_device_tf = used_device_tf
        self.context.rl_config._used_device_torch = used_device_torch

    def make_env(self) -> EnvRun:
        env = make_env(self.context.env_config)
        self.context.rl_config.setup(env)
        return env

    def make_workers(self, env: EnvRun, parameter: RLParameter, memory: RLMemory) -> List[WorkerRun]:
        # 初期化されていない場合、一人目はNone、二人目以降はrandomにする
        players = []
        for i in range(env.player_num):
            if i < len(self.context.players):
                players.append(self.context.players[i])
            else:
                if i == 0:
                    players.append(None)
                else:
                    players.append("random")

        # --- make workers
        workers = []
        for worker_type in players:
            # --- none はベース
            if worker_type is None:
                worker = make_worker(
                    self.context.rl_config,
                    env,
                    parameter,
                    memory,
                    self.context.distributed,
                    self.context.actor_id,
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
                worker = env.make_worker(
                    worker_type,
                    enable_raise=False,
                    distributed=self.context.distributed,
                    actor_id=self.context.actor_id,
                )
                if worker is not None:
                    workers.append(worker)
                    continue
                worker = make_worker_rulebase(
                    worker_type,
                    env,
                    worker_kwargs=worker_kwargs,
                    distributed=self.context.distributed,
                    actor_id=self.context.actor_id,
                )
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
                    self.context.distributed,
                    self.context.actor_id,
                )
                workers.append(worker)
                continue

            raise ValueError(f"unknown worker: {worker_type}")

        return workers


class RunStateBase:
    """
    実行中に変動する変数をまとめたクラス
    Class that summarizes variables that change during execution
    """

    def to_dict(self) -> dict:
        dat: dict = convert_for_json(self.__dict__)
        return dat
