import copy
import datetime
import enum
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Union

from srl.base.define import EnvActionType, RenderModes
from srl.base.env.config import EnvConfig
from srl.base.env.env_run import EnvRun
from srl.base.rl.base import RLMemory, RLParameter, RLTrainer
from srl.base.rl.config import RLConfig
from srl.base.rl.worker_run import WorkerRun
from srl.utils.serialize import convert_for_json

from .callback import Callback, TrainerCallback

logger = logging.getLogger(__name__)


class RunNameTypes(enum.Enum):
    main = enum.auto()
    eval = enum.auto()
    actor = enum.auto()


@dataclass
class RunContext:
    """
    実行時の状態ををまとめたクラス
    A class that summarizes the runtime state
    """

    run_name: RunNameTypes = RunNameTypes.main
    distributed: bool = False

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

    def setup(self, env_config: EnvConfig, rl_config: RLConfig, base_dir: str = "tmp"):
        if self._is_setup:
            return

        self.start_date = datetime.datetime.now()

        # "YYYYMMDD_HHMMSS_EnvName_RLName"
        dir_name = self.start_date.strftime("%Y%m%d_%H%M%S")
        dir_name += f"_{env_config.name}_{rl_config.getName()}"
        dir_name = re.sub(r'[\\/:?."<>\|]', "_", dir_name)
        self.save_dir = os.path.join(base_dir, dir_name)

        self._is_setup = True

    def to_dict(self) -> dict:
        dat: dict = convert_for_json(self.__dict__)
        return dat

    def copy(self) -> "RunContext":
        return copy.deepcopy(self)


@dataclass
class RunState:
    """
    実行中に変動する変数をまとめたクラス
    Class that summarizes variables that change during execution
    """

    env: Optional[EnvRun] = None
    workers: List[WorkerRun] = field(default_factory=list)
    trainer: Optional[RLTrainer] = None
    memory: Optional[RLMemory] = None
    parameter: Optional[RLParameter] = None

    # episodes init
    elapsed_t0: float = 0
    worker_indices: List[int] = field(default_factory=list)

    # episode state
    episode_rewards_list: List[List[float]] = field(default_factory=list)
    episode_count: int = -1
    total_step: int = 0
    end_reason: str = ""
    worker_idx: int = 0
    episode_seed: Optional[int] = None
    action: EnvActionType = 0

    # other
    sync_actor: int = 0
    sync_trainer: int = 0

    # ------------

    def to_dict(self) -> dict:
        dat: dict = convert_for_json(self.__dict__)
        return dat
