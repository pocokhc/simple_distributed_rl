import enum
import logging
import pickle
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Union

from srl.base.define import PlayerType, RenderModes
from srl.utils.serialize import convert_for_json

if TYPE_CHECKING:
    from srl.base.env.config import EnvConfig
    from srl.base.rl.config import RLConfig

logger = logging.getLogger(__name__)


class RunNameTypes(enum.Enum):
    main = enum.auto()
    trainer = enum.auto()
    actor = enum.auto()
    eval = enum.auto()


@dataclass
class RunContext:
    """
    実行時の状態をまとめたクラス
    A class that summarizes the runtime state
    """

    env_config: Optional["EnvConfig"] = None  # type: ignore , type
    rl_config: Optional["RLConfig"] = None  # type: ignore , type
    players: List[PlayerType] = field(default_factory=list)

    # --- runtime context
    run_name: RunNameTypes = RunNameTypes.main
    flow_mode: str = ""
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
    train_only: bool = False
    rollout: bool = False
    rendering: bool = False
    render_mode: Union[str, RenderModes] = RenderModes.none

    # --- thread
    enable_train_thread: bool = False
    thread_queue_capacity: int = 10

    # --- mp
    actor_id: int = 0
    actor_num: int = 1
    actor_devices: Union[str, List[str]] = "CPU"

    # --- memory
    #: None : not change
    #: <=0  : auto
    #: int  : 指定サイズで設定
    memory_limit: Optional[int] = -1

    # --- stats
    enable_stats: bool = False

    # --- random
    seed: Optional[int] = None
    seed_enable_gpu: bool = False

    # --- device option
    device: str = "AUTO"
    enable_tf_device: bool = True
    set_CUDA_VISIBLE_DEVICES_if_CPU: bool = True
    tf_device_enable: bool = True
    tf_enable_memory_growth: bool = True

    # --- device result
    framework: str = ""
    used_device_tf: str = "/CPU"
    used_device_torch: str = "cpu"

    def __post_init__(self):
        self.env_config: EnvConfig = self.env_config  # change type
        self.rl_config: RLConfig = self.rl_config  # change type

    def to_dict(self, include_env_config: bool = True, include_rl_config: bool = True) -> dict:
        d = convert_for_json(self.__dict__)
        if not include_env_config:
            del d["env_config"]
        if not include_rl_config:
            del d["rl_config"]
        return d

    def copy(self) -> "RunContext":
        context = self.__class__()
        for k, v in self.__dict__.items():
            if k in ["env_config", "rl_config"]:
                continue
            try:
                setattr(context, k, pickle.loads(pickle.dumps(v)))
            except TypeError as e:
                logger.warning(f"'{k}' copy fail.({e})")
        context.env_config = self.env_config
        context.rl_config = self.rl_config
        return context

    def check_stop_config(self):
        if self.distributed:
            assert self.max_train_count > 0 or self.timeout > 0, "Please specify 'max_train_count' or 'timeout'."
        elif self.train_only:
            assert self.max_train_count > 0 or self.timeout > 0, "Please specify 'max_train_count' or 'timeout'."
        else:
            assert (
                self.max_steps > 0  # 改行抑制コメント
                or self.max_episodes > 0
                or self.timeout > 0
                or self.max_train_count > 0
                or self.max_memory > 0
            ), "Please specify 'max_episodes', 'timeout' , 'max_steps' or 'max_train_count' or 'max_memory'."
            if self.max_memory > 0:
                if hasattr(self.rl_config, "memory"):
                    assert self.max_memory <= self.rl_config.memory_capacity
                if hasattr(self.rl_config, "memory_capacity"):
                    assert self.max_memory <= self.rl_config.memory_capacity

    def get_name(self) -> str:
        if self.run_name == RunNameTypes.actor:
            return f"actor{self.actor_id}"
        else:
            return self.run_name.name

    def set_memory_limit(self):
        from srl.base.system.memory import set_memory_limit

        set_memory_limit(self.memory_limit)

    def set_device(self):
        if self.rl_config is None:
            logger.warning("skip set device (RLConfig is None)")
            return

        self.framework = self.rl_config.get_framework()
        if self.framework == "":
            return

        from srl.base.system.device import setup_device

        used_device_tf, used_device_torch = setup_device(
            self.framework,
            self.get_device(),
            self.set_CUDA_VISIBLE_DEVICES_if_CPU,
            self.tf_enable_memory_growth,
            log_prefix=f"[{self.get_name()}]",
        )
        self.used_device_tf = used_device_tf
        self.used_device_torch = used_device_torch
        self.rl_config._used_device_tf = used_device_tf
        self.rl_config._used_device_torch = used_device_torch

    def get_device(self) -> str:
        if self.run_name == RunNameTypes.main or self.run_name == RunNameTypes.trainer:
            device = self.device.upper()
            if device == "":
                device = "AUTO"
        elif self.run_name == RunNameTypes.actor:
            if isinstance(self.actor_devices, str):
                device = self.actor_devices.upper()
            else:
                device = self.actor_devices[self.actor_id].upper()
            if device in ["", "AUTO"]:
                device = "CPU"
        else:
            device = "CPU"
        return device
