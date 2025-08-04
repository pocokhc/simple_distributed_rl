import logging
import pickle
import pprint
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Union, cast

from srl.base.define import PlayersType, RenderModeType
from srl.base.run.callback import RunCallback
from srl.base.system.device import setup_device
from srl.base.system.memory import set_memory_limit
from srl.utils.common import is_package_installed
from srl.utils.serialize import apply_dict_to_dataclass, dataclass_to_dict, get_modified_fields, load_dict, save_dict

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

    # この3つはplay前に代入する事
    env_config: "EnvConfig" = None  # type: ignore
    rl_config: "RLConfig" = None  # type: ignore
    callbacks: List[RunCallback] = field(default_factory=list)

    players: PlayersType = field(default_factory=list)

    # --- runtime context
    run_name: Literal["main", "trainer", "actor", "eval"] = "main"
    play_mode: str = ""
    # stop config
    max_episodes: int = 0
    timeout: float = 0
    max_steps: int = 0
    max_train_count: int = 0
    max_memory: int = 0
    # play config
    shuffle_player: bool = True
    disable_trainer: bool = False
    # train option
    train_interval: int = 1
    train_repeat: int = 1
    # play info
    distributed: bool = False
    training: bool = False
    train_only: bool = False
    rollout: bool = False
    # render
    env_render_mode: RenderModeType = ""
    rl_render_mode: RenderModeType = ""

    # --- mp
    actor_num: int = 1
    actor_devices: Union[str, List[str]] = "CPU"

    # --- memory
    #: None : not change
    #: <=0  : auto
    #: int  : 指定サイズで設定
    memory_limit: Optional[int] = -1

    # --- stats
    enable_stats: bool = True

    # --- random
    seed: Optional[int] = None
    seed_enable_gpu: bool = False

    # --- device option
    device: str = "AUTO"
    enable_tf_device: bool = True
    set_CUDA_VISIBLE_DEVICES_if_CPU: bool = True
    tf_device_enable: bool = True
    tf_enable_memory_growth: bool = True

    # --- private(static class instance)
    __setup_process = False

    def __post_init__(self):
        # --- mp
        self.actor_id: int = 0

        # --- device result
        self.framework: str = ""
        self.used_device_tf: str = "/CPU"
        self.used_device_torch: str = "cpu"

    def setup(self, check_stop_config: bool = True):
        assert self.env_config is not None
        assert self.rl_config is not None
        assert self.callbacks is not None

        # --- check stop config
        if check_stop_config:
            if self.distributed:
                if self.run_name == "trainer":
                    assert self.max_train_count > 0 or self.timeout > 0, "Specify one of the following: 'max_train_count', 'timeout'"
            elif self.train_only:
                assert self.max_train_count > 0 or self.timeout > 0, "Specify one of the following: 'max_train_count', 'timeout'"
            elif self.training:
                assert (
                    self.max_steps > 0  #
                    or self.max_episodes > 0
                    or self.timeout > 0
                    or self.max_train_count > 0
                    or self.max_memory > 0
                ), "Specify one of the following: 'max_episodes', 'timeout', 'max_steps', 'max_train_count', 'max_memory'"
                if self.max_memory > 0:
                    if hasattr(self.rl_config, "memory"):
                        if hasattr(self.rl_config.memory, "capacity"):  # type: ignore
                            assert self.max_memory <= self.rl_config.memory.capacity  # type: ignore
            else:
                assert self.max_steps > 0 or self.max_episodes > 0 or self.timeout > 0, "Specify one of the following: 'max_episodes', 'timeout', 'max_steps'"

        # --- setup rl config check
        assert self.rl_config.is_setup()

        # --- setup process
        self.setup_process()

    def setup_process(self):
        if not RunContext.__setup_process:
            logger.info("setup process")

            # --- memory limit
            set_memory_limit(self.memory_limit)

            # --- device
            self.setup_device()

            RunContext.__setup_process = True

    def setup_device(self, is_mp_main_process: Optional[bool] = None):
        if self.rl_config is None:
            logger.warning("skip set device (RLConfig is None)")
            return

        self.framework = self.rl_config.get_framework()
        if self.framework == "":
            return

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
        self.rl_config.set_used_device(used_device_tf, used_device_torch)

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

    # --------------------------------

    @classmethod
    def load(cls, path_or_cfg_dict: Union[dict, Any, str]) -> "RunContext":
        d = load_dict(path_or_cfg_dict)
        struct_backup = None
        if is_package_installed("omegaconf"):
            from omegaconf import OmegaConf
            from omegaconf.base import Container

            if isinstance(d, Container):
                struct_backup = OmegaConf.is_struct(cast(Container, d))
                OmegaConf.set_struct(d, False)

        # 一部特殊処理でkeyをrename
        for env_key in ["env", "envs"]:
            if env_key in d:
                d["env_config"] = d.pop(env_key)
            if "env_config" in d:
                if "_target_" not in d["env_config"]:
                    d["env_config"]["_target_"] = "srl.base.env.config.EnvConfig"
                break
        for rl_key in ["rl", "algorithm"]:
            if rl_key in d:
                d["rl_config"] = d.pop(rl_key)
                break

        # contextは展開
        for c_key in ["context", "runner"]:
            if c_key in d:
                for k, v in d.pop(c_key).items():
                    d[k] = v
                break

        # callbacksはリスト化
        if "callbacks" in d:
            if not isinstance(d["callbacks"], list):
                d["callbacks"] = [v for k, v in d["callbacks"].items()]

        if struct_backup is not None:
            OmegaConf.set_struct(cast(Container, d), struct_backup)

        return apply_dict_to_dataclass(cls(), d)

    def save(self, path: str):
        save_dict(self.to_dict(), path)
        return self

    def to_dict(self, to_print: bool = False) -> dict:
        return dataclass_to_dict(self, ["env_config", "rl_config"], to_print=to_print)

    def copy(self, copy_callbacks: bool = True) -> "RunContext":
        c = RunContext.load(dataclass_to_dict(self, ["players", "env_config", "rl_config", "callbacks"]))
        c.players = pickle.loads(pickle.dumps(self.players))
        c.env_config = self.env_config
        c.rl_config = self.rl_config
        if copy_callbacks:
            c.callbacks = self.callbacks
        return c

    def get_name(self) -> str:
        if self.run_name == "actor":
            return f"actor{self.actor_id}"
        else:
            return self.run_name

    def summary(self, show_changed_only: bool = False):
        if show_changed_only:
            d = get_modified_fields(self, ["env_config", "rl_config"])
        else:
            d = dataclass_to_dict(self, ["env_config", "rl_config"], to_print=True)
        print("--- Context ---\n" + pprint.pformat(d))
        if self.env_config is not None:
            self.env_config.summary(show_changed_only)
        if self.rl_config is not None:
            self.rl_config.summary(show_changed_only)


def load_context(path_or_cfg_dict: Union[dict, Any, str]) -> RunContext:
    return RunContext.load(path_or_cfg_dict)


@dataclass
class RunState:
    """
    実行中の状態をまとめたクラス
    A class that summarizes the execution state
    """

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
    action: Any = None
    train_count: int = 0

    # train
    is_step_trained: bool = False

    # distributed
    sync_actor: int = 0
    actor_send_q: int = 0
    sync_trainer: int = 0
    trainer_recv_q: int = 0

    # info(簡単な情報はここに保存)
    last_episode_step: float = 0
    last_episode_time: float = 0
    last_episode_rewards: List[float] = field(default_factory=list)

    # other
    shared_vars: dict = field(default_factory=dict)


@dataclass
class RunStateActor(RunState):
    env: "EnvRun" = None  # type: ignore
    worker: "WorkerRun" = None  # type: ignore
    workers: List["WorkerRun"] = None  # type: ignore
    parameter: "RLParameter" = None  # type: ignore
    memory: "RLMemory" = None  # type: ignore
    trainer: Optional["RLTrainer"] = None


@dataclass
class RunStateTrainer(RunState):
    trainer: "RLTrainer" = None  # type: ignore
    memory: "RLMemory" = None  # type: ignore
    parameter: "RLParameter" = None  # type: ignore
