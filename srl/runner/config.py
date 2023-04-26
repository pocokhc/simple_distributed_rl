import enum
import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

import srl
import srl.rl.dummy
import srl.rl.human  # reservation
import srl.rl.random_play  # reservation
from srl.base.define import PlayRenderMode
from srl.base.env.base import EnvRun
from srl.base.env.config import EnvConfig
from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory, RLTrainer
from srl.base.rl.registration import (
    make_parameter,
    make_remote_memory,
    make_trainer,
    make_worker,
    make_worker_rulebase,
)
from srl.base.rl.worker import WorkerRun
from srl.runner.callback import Callback
from srl.utils.common import is_available_gpu_tf, is_package_imported

logger = logging.getLogger(__name__)


@dataclass
class Config:
    env_config: Union[str, EnvConfig]
    rl_config: Optional[RLConfig]

    # random
    seed: Optional[int] = None
    seed_enable_gpu: bool = True  # 有効にならない場合あり、また速度が犠牲になる可能性あり

    # multi player option
    # playersという変数名だけど、役割はworkersの方が正しい
    players: List[Union[None, str, RLConfig]] = field(default_factory=list)

    # mp options
    actor_num: int = 1
    trainer_parameter_send_interval_by_train_count: int = 100
    actor_parameter_sync_interval_by_step: int = 100

    """ device option
    "AUTO",""     : Automatic assignment.
    "CPU","CPU:0" : Use CPU.
    "GPU","GPU:0" : Use GPU.

    AUTO assign
    - sequence
        - device  : GPU > CPU
        - trainer : not use
        - actors  : not use
    - distribute
        - device  : CPU
        - trainer : GPU > CPU
        - actors  : CPU
    """
    device: str = "AUTO"
    device_mp_trainer: str = "AUTO"
    device_mp_actor: Union[str, List[str]] = "AUTO"  # ["CPU:0", "CPU:1"]
    on_device_init_function: Optional[Callable[["Config"], None]] = None
    use_CUDA_VISIBLE_DEVICES: bool = True  # CPUの場合 CUDA_VISIBLE_DEVICES を-1にする

    # tensorflow options
    tf_disable: bool = False
    tf_enable_memory_growth: bool = True

    def __post_init__(self):
        # stop config
        self.max_episodes: int = -1
        self.timeout: int = -1
        self.max_steps: int = -1
        self.max_train_count: int = -1
        # play config
        self.shuffle_player: bool = False
        self.disable_trainer: bool = False
        self.render_mode: PlayRenderMode = PlayRenderMode.none
        self.render_kwargs: dict = {}
        self.enable_profiling: bool = True
        # callbacks
        self.callbacks: List[Callback] = []

        # play info
        self.training: bool = False
        self.distributed: bool = False
        self.enable_ps: bool = False
        self.enable_nvidia: bool = False
        self.run_name: str = "main"
        self.run_actor_id: int = 0

        # The device used by the framework.
        self.used_device_tf: str = "/CPU"
        self.used_device_torch: str = "cpu"
        self.__is_init_device: bool = False

        if self.rl_config is None:
            self.rl_config = srl.rl.dummy.Config()
        if isinstance(self.env_config, str):
            self.env_config = EnvConfig(self.env_config)

        self.rl_name = self.rl_config.getName()
        self.env = None

    # ------------------------------
    # user functions
    # ------------------------------
    def model_summary(self, **kwargs) -> RLParameter:
        self.make_env()
        parameter = self.make_parameter()
        parameter.summary(**kwargs)
        return parameter

    # ------------------------------
    # runner functions
    # ------------------------------
    def assert_params(self):
        self.make_env()
        assert self.actor_num > 0
        self.rl_config.assert_params()

    def _set_env(self):
        if self.env is not None:
            return
        self.env = srl.make_env(self.env_config)
        self.rl_config.reset(self.env)

        # 初期化されていない場合、一人目はNone、二人目以降はrandomにする
        if len(self.players) == 0:
            self.players = ["random" for _ in range(self.env.player_num)]
            self.players[0] = None

    def make_env(self) -> EnvRun:
        self.init_device()
        self._set_env()
        self.env.init()
        return self.env

    def make_parameter(self, is_load: bool = True) -> RLParameter:
        self.init_device()
        self._set_env()
        return make_parameter(self.rl_config, env=self.env, is_load=is_load)

    def make_remote_memory(self, is_load: bool = True) -> RLRemoteMemory:
        self.init_device()
        self._set_env()
        return make_remote_memory(self.rl_config, env=self.env, is_load=is_load)

    def make_trainer(self, parameter: RLParameter, remote_memory: RLRemoteMemory) -> RLTrainer:
        self.init_device()
        self._set_env()
        return make_trainer(self.rl_config, parameter, remote_memory, env=self.env)

    def make_worker(
        self,
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
        actor_id: int = 0,
    ) -> WorkerRun:
        self.init_device()
        self._set_env()
        worker = make_worker(
            self.rl_config,
            parameter,
            remote_memory,
            self.env,
            self.training,
            self.distributed,
            actor_id,
        )
        return worker

    def make_player(
        self,
        player: Union[None, str, RLConfig],
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
        actor_id: int = 0,
        env_worker_kwargs={},
    ) -> WorkerRun:
        self.init_device()
        env = self.make_env()

        # none はベース
        if player is None:
            return self.make_worker(parameter, remote_memory, actor_id)

        # 文字列はenv側またはルールベースのアルゴリズム
        if isinstance(player, str):
            worker = env.make_worker(
                player,
                self.training,
                self.distributed,
                enable_raise=False,
                env_worker_kwargs=env_worker_kwargs,
            )
            if worker is not None:
                return worker
            worker = make_worker_rulebase(
                player,
                self.training,
                self.distributed,
            )
            if worker is not None:
                return worker
            assert False, f"not registered: {player}"

        # RLConfigは専用のWorkerを作成
        if isinstance(player, object) and issubclass(player.__class__, RLConfig):
            parameter = make_parameter(self.rl_config)
            remote_memory = make_remote_memory(self.rl_config)
            worker = make_worker(
                player,
                parameter,
                remote_memory,
                env,
                self.training,
                self.distributed,
                actor_id,
            )
            return worker

        raise ValueError(f"unknown worker: {player}")

    # ------------------------------
    # GPU
    # ------------------------------
    def get_device_name(self) -> str:
        if self.run_name in ["main", "eval"]:
            device = self.device
            if device in ["", "AUTO"]:
                if self.distributed:
                    device = "CPU"
                else:
                    device = "AUTO"
        elif self.run_name == "trainer":
            device = self.device_mp_trainer
            if device == "":
                device = "AUTO"
        elif "actor" in self.run_name:
            if isinstance(self.device_mp_actor, str):
                device = self.device_mp_actor
            else:
                device = self.device_mp_actor[self.run_actor_id]
            if device in ["", "AUTO"]:
                device = "CPU"
        else:
            raise ValueError("not coming")

        return device

    def init_device(self):
        if self.__is_init_device:
            return

        if "CUDA_VISIBLE_DEVICES" in os.environ:
            cuda_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            logger.info(f"[{self.run_name}] CUDA_VISIBLE_DEVICES='{cuda_devices}'")
        else:
            logger.info(f"[{self.run_name}] CUDA_VISIBLE_DEVICES is not define.")

        # --- device
        device = self.get_device_name()
        logger.info(f"[{self.run_name}] config device name: {device}")

        # --- memory growth, Tensorflow,GPU がある場合に実施(CPUにしてもなぜかGPUの初期化は走る場合あり)
        if self.tf_enable_memory_growth and is_package_imported("tensorflow") and is_available_gpu_tf():
            import tensorflow as tf

            try:
                gpu_devices = tf.config.list_physical_devices("GPU")
                for d in gpu_devices:
                    logger.info(f"[{self.run_name}] (tf) set_memory_growth({d.name}, True)")
                    tf.config.experimental.set_memory_growth(d, True)
            except Exception:
                s = f"[{self.run_name}] (tf) 'set_memory_growth' failed."
                s += " Also consider 'tf_enable_memory_growth=False'."
                print(s)
                raise

        if "CPU" in device:
            if self.use_CUDA_VISIBLE_DEVICES:
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                logger.info(f"[{self.run_name}] set CUDA_VISIBLE_DEVICES=-1")

            if "CPU" == device:
                self.used_device_tf = "/CPU"
                self.used_device_torch = "cpu"
            elif "CPU:" in device:
                t = device.split(":")
                self.used_device_tf = f"/CPU:{t[1]}"
                self.used_device_torch = f"cpu:{t[1]}"
        else:
            # CUDA_VISIBLE_DEVICES が -1 の場合のみ削除する
            if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "-1":
                del os.environ["CUDA_VISIBLE_DEVICES"]
                logger.info(f"[{self.run_name}] del CUDA_VISIBLE_DEVICES")

            # --- tensorflow GPU check
            if is_package_imported("tensorflow") or (self.rl_config.get_use_framework() == "tensorflow"):
                import tensorflow as tf

                gpu_devices = tf.config.list_physical_devices("GPU")
                if len(gpu_devices) == 0:
                    assert (
                        "GPU" not in device
                    ), f"[{self.run_name}] (tf) GPU is not found. {tf.config.list_physical_devices()}"

                    self.used_device_tf = "/CPU"

                else:
                    logger.info(f"[{self.run_name}] (tf) gpu device: {len(gpu_devices)}")

                    if "GPU:" in device:
                        t = device.split(":")
                        self.used_device_tf = f"/GPU:{t[1]}"
                    else:
                        self.used_device_tf = "/GPU"

            # --- torch GPU check
            if is_package_imported("torch") or (self.rl_config.get_use_framework() == "torch"):
                import torch

                if not torch.cuda.is_available():
                    assert (
                        "GPU" not in device
                    ), f"[{self.run_name}] (torch) GPU is not found. {tf.config.list_physical_devices()}"

                    self.used_device_torch = "cpu"
                else:
                    logger.info(f"[{self.run_name}] (torch) gpu device: {torch.cuda.get_device_name()}")

                    if "GPU:" in device:
                        t = device.split(":")
                        self.used_device_torch = f"cuda:{t[1]}"
                    else:
                        self.used_device_torch = "cuda"

        if self.on_device_init_function is not None:
            self.on_device_init_function(self)

        if is_package_imported("tensorflow") or (self.rl_config.get_use_framework() == "tensorflow"):
            logger.info(f"[{self.run_name}] The device used by Tensorflow '{self.used_device_tf}'.")
        if is_package_imported("torch") or (self.rl_config.get_use_framework() == "torch"):
            logger.info(f"[{self.run_name}] The device used by Torch '{self.used_device_torch}'.")

        self.rl_config.used_device_tf = self.used_device_tf
        self.rl_config.used_device_torch = self.used_device_torch
        self.env_config.used_device_tf = self.used_device_tf
        self.env_config.used_device_torch = self.used_device_torch
        self.__is_init_device = True
        logger.info(f"[{self.run_name}] Initialized device.")

    def on_init_process(self):
        """プロセスの最初に実行される"""
        self.__is_init_device = False
        self.init_device()

    # ------------------------------
    # other functions
    # ------------------------------
    def to_dict(self) -> dict:
        conf = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if v is None or type(v) in [int, float, bool, str]:
                conf[k] = v
            elif type(v) is list:
                conf[k] = [str(n) for n in v]
            elif type(v) is dict:
                conf[k] = v.copy()
            elif issubclass(type(v), enum.Enum):
                conf[k] = v.name
            else:
                conf[k] = str(v)

        conf["rl_config"] = {}
        for k, v in self.rl_config.__dict__.items():
            if k.startswith("_"):
                continue
            if v is None or type(v) in [int, float, bool, str]:
                conf["rl_config"][k] = v
            elif type(v) is list:
                conf["rl_config"][k] = [str(n) for n in v]
            elif type(v) is dict:
                conf["rl_config"][k] = v.copy()
            elif issubclass(type(v), enum.Enum):
                conf["rl_config"][k] = v.name
            else:
                conf["rl_config"][k] = str(v)

        conf["env_config"] = {}
        for k, v in self.env_config.__dict__.items():
            if k.startswith("_"):
                continue
            if v is None or type(v) in [int, float, bool, str]:
                conf["env_config"][k] = v
            elif type(v) is list:
                conf["env_config"][k] = [str(n) for n in v]
            elif type(v) is dict:
                conf["env_config"][k] = v.copy()
            elif issubclass(type(v), enum.Enum):
                conf["env_config"][k] = v.name
            else:
                conf["env_config"][k] = str(v)

        return conf

    def copy(self, env_share: bool = False, callbacks_share: bool = True):
        self._set_env()

        env_config = self.env_config.copy()
        rl_config = self.rl_config.copy()
        config = Config(env_config, rl_config)

        # parameter
        for k, v in self.__dict__.items():
            if v is None or type(v) in [int, float, bool, str]:
                setattr(config, k, v)
            elif issubclass(type(v), enum.Enum):
                setattr(config, k, v)

        # list parameter
        config.players = []
        for player in self.players:
            if player is None:
                config.players.append(None)
            else:
                config.players.append(pickle.loads(pickle.dumps(player)))

        # callback
        if callbacks_share:
            config.callbacks = self.callbacks
        else:
            config.callbacks = pickle.loads(pickle.dumps(self.callbacks))

        # env
        if env_share:
            config.env = self.env

        return config

    # ------------------------------
    # utility
    # ------------------------------
    def get_env_init_state(self, encode: bool = True) -> np.ndarray:
        env = self.make_env()
        env.reset()
        state = env.state
        if encode:
            worker = self.make_worker()
            state = worker.worker.state_encode(state, env)
        return state


def save(
    path: str, config: Config, parameter: Optional[RLParameter] = None, remote_memory: Optional[RLRemoteMemory] = None
) -> None:
    dat = [
        config,
        parameter.backup() if parameter is not None else None,
        remote_memory.backup(compress=True) if remote_memory is not None else None,
    ]
    with open(path, "wb") as f:
        pickle.dump(dat, f)


def load(path: str) -> Tuple[Config, RLParameter, RLRemoteMemory]:
    with open(path, "rb") as f:
        dat = pickle.load(f)
    config = dat[0]
    parameter = config.make_parameter()
    if dat[1] is not None:
        parameter.restore(dat[1])
    remote_memory = config.make_remote_memory()
    if dat[2] is not None:
        remote_memory.restore(dat[2])
    return config, parameter, remote_memory
