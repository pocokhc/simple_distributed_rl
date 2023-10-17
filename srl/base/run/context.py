import copy
import datetime
import enum
import logging
import os
import pickle
import re
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

from .callback import Callback, TrainerCallback

logger = logging.getLogger(__name__)


class RunNameTypes(enum.Enum):
    main = enum.auto()
    trainer = enum.auto()
    actor = enum.auto()
    eval = enum.auto()


class TrainingModeTypes(enum.Enum):
    short = enum.auto()
    long = enum.auto()


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

    run_name: RunNameTypes = RunNameTypes.main
    #: 短い場合はwkdirを分けて作成
    #: 長い場合はwkdirを同じにして、実行をまたいでリストアさせる
    training_mode: TrainingModeTypes = TrainingModeTypes.short

    # --- mp
    actor_id: int = 0
    actor_num: int = 1
    trainer_parameter_send_interval_by_train_count: int = 100
    actor_parameter_sync_interval_by_step: int = 100
    enable_prepare_batch: bool = False

    # --- play context
    # stop config
    max_episodes: int = 0
    timeout: int = 0
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

    # --- random
    seed: Optional[int] = None

    # --- device
    framework: str = ""
    enable_tf_device: bool = True
    used_device_tf: str = "/CPU"
    used_device_torch: str = "cpu"

    # --- callbacks
    callbacks: List[Union[Callback, TrainerCallback]] = field(default_factory=list)

    def __post_init__(self):
        self._is_setup = False

        self.wkdir: str = ""
        self.start_date = datetime.datetime(2000, 1, 1)

    def create_controller(self) -> "RunContextController":
        return RunContextController(self)


class RunContextController:
    def __init__(self, context: RunContext):
        self.context = context

    def to_dict(self) -> dict:
        dat: dict = convert_for_json(self.context.__dict__)
        return dat

    def copy(self, copy_setup: bool, copy_callbacks: bool) -> "RunContext":
        _c = self.context.callbacks

        self.context.callbacks = []
        c = copy.deepcopy(self.context)
        self.context.callbacks = _c

        if copy_callbacks:
            c.callbacks = _c

        if copy_setup:
            c._is_setup = self.context._is_setup
        else:
            c._is_setup = False
        return c

    def setup(self, training_mode: TrainingModeTypes = TrainingModeTypes.short, wkdir: str = "tmp"):
        if self.context._is_setup:
            return

        self.context.training_mode = training_mode

        if training_mode == TrainingModeTypes.short:
            self.context.start_date = datetime.datetime.now()

            # "YYYYMMDD_HHMMSS_EnvName_RLName"
            dir_name = self.context.start_date.strftime("%Y%m%d_%H%M%S")
            dir_name += f"_{self.context.env_config.name}_{self.context.rl_config.getName()}"
            dir_name = re.sub(r'[\\/:?."<>\|]', "_", dir_name)
            self.context.wkdir = os.path.join(wkdir, dir_name)

        elif training_mode == TrainingModeTypes.long:
            raise NotImplementedError("TODO")
        else:
            raise ValueError(training_mode)

        self.context._is_setup = True

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
