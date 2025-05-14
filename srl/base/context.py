import logging
import pickle
import pprint
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Union

from srl.base.define import PlayersType, RenderModeType
from srl.utils.serialize import convert_for_json

if TYPE_CHECKING:
    from srl.base.env.config import EnvConfig
    from srl.base.env.env_run import EnvRun
    from srl.base.rl.config import RLConfig
    from srl.base.rl.memory import RLMemory
    from srl.base.rl.parameter import RLParameter
    from srl.base.rl.trainer import RLTrainer
    from srl.base.rl.worker_run import WorkerRun

logger = logging.getLogger(__name__)


@dataclass
class RunContext:
    """
    実行時の状態をまとめたクラス
    A class that summarizes the runtime state
    """

    env_config: Optional["EnvConfig"] = None  # type: ignore , type
    rl_config: Optional["RLConfig"] = None  # type: ignore , type
    players: PlayersType = field(default_factory=list)

    # --- runtime context
    run_name: Literal["main", "trainer", "actor", "eval"] = "main"
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
    # render
    env_render_mode: RenderModeType = ""
    rl_render_mode: RenderModeType = ""

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
        context.env_config = self.env_config.copy()
        context.rl_config = self.rl_config.copy()
        return context

    def check_stop_config(self):
        if self.distributed:
            if self.run_name == "trainer":
                assert self.max_train_count > 0 or self.timeout > 0, "Please specify 'max_train_count' or 'timeout'."
        elif self.train_only:
            assert self.max_train_count > 0 or self.timeout > 0, "Please specify 'max_train_count' or 'timeout'."
        else:
            assert (
                self.max_steps > 0  #
                or self.max_episodes > 0
                or self.timeout > 0
                or self.max_train_count > 0
                or self.max_memory > 0
            ), "Please specify 'max_episodes', 'timeout' , 'max_steps' or 'max_train_count' or 'max_memory'."
            if self.max_memory > 0:
                if hasattr(self.rl_config, "memory"):
                    if hasattr(self.rl_config.memory, "capacity"):  # type: ignore
                        assert self.max_memory <= self.rl_config.memory.capacity  # type: ignore

    def get_name(self) -> str:
        if self.run_name == "actor":
            return f"actor{self.actor_id}"
        else:
            return self.run_name

    def setup_rl_config(self):
        if (self.rl_config is not None) and (not self.rl_config.is_setup()):
            self.rl_config.setup(self.env_config.make())

    def set_memory_limit(self):
        from srl.base.system.memory import set_memory_limit

        set_memory_limit(self.memory_limit)

    def setup_device(self, is_mp_main_process: Optional[bool] = None):
        if self.rl_config is None:
            logger.warning("skip set device (RLConfig is None)")
            return

        self.framework = self.rl_config.get_framework()
        if self.framework == "":
            return

        from srl.base.system.device import setup_device

        tf_policy_name = "mixed_float16" if self.rl_config.dtype == "float16" else ""
        used_device_tf, used_device_torch = setup_device(
            self.framework,
            self.get_device(),
            is_mp_main_process,
            self.set_CUDA_VISIBLE_DEVICES_if_CPU,
            self.tf_enable_memory_growth,
            tf_policy_name,
            log_prefix=f"[{self.get_name()}]",
        )
        self.used_device_tf = used_device_tf
        self.used_device_torch = used_device_torch
        self.rl_config._used_device_tf = used_device_tf
        self.rl_config._used_device_torch = used_device_torch

    def get_device(self) -> str:
        if self.run_name == "main" or self.run_name == "trainer":
            device = self.device.upper()
            if device == "":
                device = "AUTO"
        elif self.run_name == "actor":
            if isinstance(self.actor_devices, str):
                device = self.actor_devices.upper()
            else:
                device = self.actor_devices[self.actor_id].upper()
            if device in ["", "AUTO"]:
                device = "CPU"
        else:
            device = "CPU"
        return device

    def to_str_context(self, include_env_config: bool = True, include_rl_config: bool = True, include_context: bool = True) -> str:
        s = ""
        if include_env_config and (self.env_config is not None):
            s += "--- EnvConfig ---\n" + pprint.pformat(self.env_config.to_dict())
        if include_rl_config and (self.rl_config is not None):
            s += "--- RLConfig ---\n" + pprint.pformat(self.rl_config.to_dict())
        if include_context:
            s += "--- Context ---\n" + pprint.pformat(self.to_dict(include_env_config=False, include_rl_config=False))
        return s

    def print_context(self, include_env_config: bool = True, include_rl_config: bool = True, include_context: bool = True):
        print(self.to_str_context(include_env_config, include_rl_config, include_context))


class RunState:
    """
    実行中の状態をまとめたクラス
    A class that summarizes the execution state
    """

    def __init__(self) -> None:
        self.env: Optional["EnvRun"] = None
        self.worker: Optional["WorkerRun"] = None  # main worker
        self.workers: Optional[List["WorkerRun"]] = None
        self.memory: Optional["RLMemory"] = None
        self.parameter: Optional["RLParameter"] = None
        self.trainer: Optional["RLTrainer"] = None
        self.init()

    def init(self):
        # episodes init
        self.elapsed_t0: float = 0
        self.worker_indices: List[int] = []

        # episode state
        self.episode_rewards_list: List[List[float]] = []
        self.episode_count: int = -1
        self.total_step: int = 0
        self.end_reason: str = ""
        self.worker_idx: int = 0
        self.episode_seed: Optional[int] = None
        self.action: Any = None
        self.train_count: int = 0

        # train
        self.is_step_trained: bool = False

        # distributed
        self.sync_actor: int = 0
        self.actor_send_q: int = 0
        self.sync_trainer: int = 0
        self.trainer_recv_q: int = 0

        # info(簡単な情報はここに保存)
        self.last_episode_step: float = 0
        self.last_episode_time: float = 0
        self.last_episode_rewards: List[float] = []

        # other
        self.shared_vars: dict = {}

    def to_dict(self) -> dict:
        dat: dict = convert_for_json(self.__dict__)
        return dat
