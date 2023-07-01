import datetime
import enum
import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Callable, ClassVar, List, Optional, Tuple, Union

from srl.base.define import EnvObservationType, RLObservationType
from srl.base.env.config import EnvConfig
from srl.base.env.env_run import EnvRun
from srl.base.env.registration import make as srl_make_env
from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory, RLTrainer
from srl.base.rl.registration import (
    make_parameter,
    make_remote_memory,
    make_trainer,
    make_worker,
    make_worker_rulebase,
)
from srl.base.rl.worker_run import WorkerRun
from srl.rl import dummy
from srl.runner.callback import Callback

logger = logging.getLogger(__name__)


@dataclass()
class Config:
    name_or_env_config: Union[str, EnvConfig]
    rl_config: Optional[RLConfig]

    players: List[Union[None, str, Tuple[str, dict], RLConfig]] = field(default_factory=list)
    """ multi player option, playersという変数名だけど、役割はworkersの方が正しい
    None             : use rl_config worker
    str              : Registered RuleWorker
    Tuple[str, dict] : Registered RuleWorker(Pass kwargs argument)
    RLConfig         : use RLConfig worker
    """

    # --- mp options
    actor_num: int = 1
    trainer_parameter_send_interval_by_train_count: int = 100
    actor_parameter_sync_interval_by_step: int = 100

    # --- device option
    device_main: str = "AUTO"
    """ device option
    "AUTO",""    : Automatic assignment.
    "CPU","CPU:0": Use CPU.
    "GPU","GPU:0": Use GPU.

    AUTO assign
    - sequence
        - main   : GPU > CPU
        - trainer: not use
        - actors : not use
    - distribute
        - main   : CPU
        - trainer: GPU > CPU
        - actors : CPU
    """
    device_mp_trainer: str = "AUTO"
    device_mp_actors: Union[str, List[str]] = "AUTO"  # ["CPU:0", "CPU:1"]
    on_device_init_function: Optional[Callable[["Config"], None]] = None
    use_CUDA_VISIBLE_DEVICES: bool = True  # CPUの場合 CUDA_VISIBLE_DEVICES を-1にする

    # --- remote options
    remote_server_ip: str = "127.0.0.1"  # "0.0.0.0" for external publication
    remote_port: int = 50000
    remote_authkey: bytes = b"abracadabra"

    # --- tensorflow options
    tf_disable: bool = False
    tf_enable_memory_growth: bool = True

    # --- random
    seed: Optional[int] = None
    seed_enable_gpu: bool = True  # 有効にならない場合あり、また速度が犠牲になる可能性あり

    # --- other
    # 基本となるディレクトリ、ファイル関係は、この配下に時刻でフォルダが作られ、その下が基準となる
    base_dir: str = "tmp"

    # --- private
    __is_init_device: ClassVar[bool] = False

    def __post_init__(self):
        if isinstance(self.name_or_env_config, str):
            self.env_config = EnvConfig(self.name_or_env_config)
        else:
            self.env_config: EnvConfig = self.name_or_env_config
        if self.rl_config is None:
            self.rl_config = dummy.Config()
        self.rl_config: RLConfig = self.rl_config

        # stop config
        self._max_episodes: int = -1
        self._timeout: int = -1
        self._max_steps: int = -1
        self._max_train_count: int = -1
        # play config
        self._shuffle_player: bool = False
        self._disable_trainer: bool = False
        self._enable_profiling: bool = True
        # callbacks
        self._callbacks: List[Callback] = []

        # other
        self._now = datetime.datetime.now()
        self._save_dir = None

        # play info
        self._training: bool = False
        self._distributed: bool = False
        self._enable_psutil: bool = False
        self._enable_nvidia: bool = False
        self._run_name: str = "main"
        self._actor_id: int = 0

        # The device used by the framework.
        self._used_device_tf: str = "/CPU"
        self._used_device_torch: str = "cpu"

        # --------------------------------------------

        self.rl_name = self.rl_config.getName()
        self.env = None

        # import check
        self._use_tf = False
        self._use_torch = False
        framework = self.rl_config.get_use_framework()
        if framework == "tensorflow":
            self._use_tf = True
        elif framework == "torch":
            self._use_torch = True
        self.init_device()

    # --- runner内で設定する変数
    @property
    def max_episodes(self) -> int:
        return self._max_episodes

    @property
    def timeout(self) -> int:
        return self._timeout

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def max_train_count(self) -> int:
        return self._max_train_count

    @property
    def shuffle_player(self) -> bool:
        return self._shuffle_player

    @property
    def disable_trainer(self) -> bool:
        return self._disable_trainer

    @property
    def enable_profiling(self) -> bool:
        return self._enable_profiling

    @property
    def callbacks(self) -> List[Callback]:
        return self._callbacks

    @property
    def now(self) -> datetime.datetime:
        return self._now

    @property
    def save_dir(self) -> str:
        if self._save_dir is None:
            import re

            # "YYYYMMDD_HHMMSS_EnvName_RLName"
            dir_name = self._now.strftime("%Y%m%d_%H%M%S")
            dir_name += f"_{self.env_config.name}_{self.rl_config.getName()}"
            dir_name = re.sub(r'[\\/:?."<>\|]', "_", dir_name)
            self._save_dir = os.path.join(self.base_dir, dir_name)
            logger.info(f"save_dir: {os.path.abspath(self._save_dir)}")
        return self._save_dir

    @property
    def training(self) -> bool:
        return self._training

    @property
    def distributed(self) -> bool:
        return self._distributed

    @property
    def enable_psutil(self) -> bool:
        return self._enable_psutil

    @property
    def enable_nvidia(self) -> bool:
        return self._enable_nvidia

    @property
    def run_name(self) -> str:
        return self._run_name

    @property
    def actor_id(self) -> int:
        return self._actor_id

    @property
    def used_device_tf(self) -> str:
        return self._used_device_tf

    @property
    def used_device_torch(self) -> str:
        return self._used_device_torch

    @property
    def use_tf(self) -> bool:
        return self._use_tf

    @property
    def use_torch(self) -> bool:
        return self._use_torch

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
    def init_play(self):
        """runを始める最初に実行する"""
        # rl_configを初期化
        if not self.rl_config.is_set_env_config:
            self.rl_config.reset(self.make_env())

        # assert params
        self.assert_params()

    def init_process(self):
        """mp等別プロセスを起動した場合、最初に実行する"""
        Config.__is_init_device = False
        self.init_device()

    def assert_params(self):
        self.make_env()
        assert self.actor_num > 0
        self.rl_config.assert_params()

    def make_env(self) -> EnvRun:
        if self.env is None:
            self.env = srl_make_env(self.env_config)
        self.env.init()
        return self.env

    def make_parameter(self, is_load: bool = True) -> RLParameter:
        return make_parameter(self.rl_config, env=self.make_env(), is_load=is_load)

    def make_remote_memory(self, is_load: bool = True) -> RLRemoteMemory:
        return make_remote_memory(self.rl_config, env=self.make_env(), is_load=is_load)

    def make_trainer(self, parameter: RLParameter, remote_memory: RLRemoteMemory) -> RLTrainer:
        return make_trainer(self.rl_config, parameter, remote_memory, env=self.make_env())

    def make_worker(
        self,
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
    ) -> WorkerRun:
        return make_worker(
            self.rl_config,
            self.make_env(),
            parameter,
            remote_memory,
            self.distributed,
            self.actor_id,
        )

    def make_worker_player(
        self,
        player: Union[None, str, RLConfig],
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
        worker_kwargs={},
    ) -> WorkerRun:
        env = self.make_env()

        # none はベース
        if player is None:
            return self.make_worker(parameter, remote_memory)

        # 文字列はenv側またはルールベースのアルゴリズム
        if isinstance(player, str):
            worker = env.make_worker(
                player,
                self.distributed,
                enable_raise=False,
                env_worker_kwargs=worker_kwargs,
            )
            if worker is not None:
                return worker
            worker = make_worker_rulebase(
                player,
                env,
                distributed=self.distributed,
                actor_id=self.actor_id,
                update_config_parameter=worker_kwargs,
            )
            assert worker is not None, f"not registered: {player}"
            return worker

        # RLConfigは専用のWorkerを作成
        if isinstance(player, object) and issubclass(player.__class__, RLConfig):
            parameter = make_parameter(self.rl_config)
            remote_memory = make_remote_memory(self.rl_config)
            worker = make_worker(
                player,
                env,
                parameter,
                remote_memory,
                self.distributed,
                self.actor_id,
            )
            return worker

        raise ValueError(f"unknown worker: {player}")

    def make_players(
        self,
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
    ) -> List[WorkerRun]:
        env = self.make_env()

        # 初期化されていない場合、一人目はNone、二人目以降はrandomにする
        if len(self.players) == 0:
            players: List[Union[None, str, Tuple[str, dict], RLConfig]] = ["random" for _ in range(env.player_num)]
            players[0] = None
        else:
            players = self.players

        workers = []
        for i in range(env.player_num):
            p = players[i] if i < len(players) else None
            kwargs = {}
            if isinstance(p, tuple) or isinstance(p, list):
                kwargs = p[1]
                p = p[0]
            workers.append(
                self.make_worker_player(
                    p,
                    parameter,
                    remote_memory,
                    kwargs,
                )
            )
        return workers

    # ------------------------------
    # GPU
    # ------------------------------
    def get_device_name(self) -> str:
        if self.run_name in ["main", "eval"]:
            device = self.device_main.upper()
            if device in ["", "AUTO"]:
                if self.distributed:
                    device = "CPU"
                else:
                    device = "AUTO"
        elif self.run_name == "trainer":
            device = self.device_mp_trainer.upper()
            if device == "":
                device = "AUTO"
        elif "actor" in self.run_name:
            if isinstance(self.device_mp_actors, str):
                device = self.device_mp_actors.upper()
            else:
                device = self.device_mp_actors[self.actor_id].upper()
            if device in ["", "AUTO"]:
                device = "CPU"
        else:
            raise ValueError(f"not coming(run_name={self.run_name})")

        return device

    def init_device(self):
        if Config.__is_init_device:
            return

        if "CUDA_VISIBLE_DEVICES" in os.environ:
            cuda_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            logger.info(f"[{self.run_name}] CUDA_VISIBLE_DEVICES='{cuda_devices}'")
        else:
            logger.info(f"[{self.run_name}] CUDA_VISIBLE_DEVICES is not define.")

        # --- device
        device = self.get_device_name()
        logger.info(f"[{self.run_name}] config device name: {device}")

        # --- CPU
        if "CPU" in device:
            if self.use_CUDA_VISIBLE_DEVICES:
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                logger.info(f"[{self.run_name}] set CUDA_VISIBLE_DEVICES=-1")

            if "CPU" == device:
                self._used_device_tf = "/CPU"
                self._used_device_torch = "cpu"
            elif "CPU:" in device:
                t = device.split(":")
                self._used_device_tf = f"/CPU:{t[1]}"
                self._used_device_torch = f"cpu:{t[1]}"

        # --- memory growth, Tensorflow,GPU がある場合に実施(CPUにしてもなぜかGPUの初期化は走る場合あり)
        if self.tf_enable_memory_growth and self.use_tf:
            try:
                import tensorflow as tf

                gpu_devices = tf.config.list_physical_devices("GPU")
                for d in gpu_devices:
                    logger.info(f"[{self.run_name}] (tf) set_memory_growth({d.name}, True)")
                    tf.config.experimental.set_memory_growth(d, True)
            except Exception:
                s = f"[{self.run_name}] (tf) 'set_memory_growth' failed."
                s += " Also consider 'tf_enable_memory_growth=False'."
                print(s)
                raise

        # --- GPU
        if "CPU" not in device:
            # CUDA_VISIBLE_DEVICES が -1 の場合のみ削除する
            if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "-1":
                del os.environ["CUDA_VISIBLE_DEVICES"]
                logger.info(f"[{self.run_name}] del CUDA_VISIBLE_DEVICES")

            # --- tensorflow GPU check
            if self.use_tf:
                import tensorflow as tf

                gpu_devices = tf.config.list_physical_devices("GPU")
                if len(gpu_devices) == 0:
                    assert (
                        "GPU" not in device
                    ), f"[{self.run_name}] (tf) GPU is not found. {tf.config.list_physical_devices()}"

                    self._used_device_tf = "/CPU"

                else:
                    logger.info(f"[{self.run_name}] (tf) gpu device: {len(gpu_devices)}")

                    if "GPU:" in device:
                        t = device.split(":")
                        self._used_device_tf = f"/GPU:{t[1]}"
                    else:
                        self._used_device_tf = "/GPU"

            # --- torch GPU check
            if self.use_torch:
                import torch

                if torch.cuda.is_available():
                    logger.info(f"[{self.run_name}] (torch) gpu device: {torch.cuda.get_device_name()}")

                    if "GPU:" in device:
                        t = device.split(":")
                        self._used_device_torch = f"cuda:{t[1]}"
                    else:
                        self._used_device_torch = "cuda"
                else:
                    assert "GPU" not in device, f"[{self.run_name}] (torch) GPU is not found."

                    self._used_device_torch = "cpu"

        if self.on_device_init_function is not None:
            self.on_device_init_function(self)

        if self.use_tf:
            logger.info(f"[{self.run_name}] The device used by Tensorflow '{self.used_device_tf}'.")
        if self.use_torch:
            logger.info(f"[{self.run_name}] The device used by Torch '{self.used_device_torch}'.")

        self.rl_config._used_device_tf = self.used_device_tf
        self.rl_config._used_device_torch = self.used_device_torch
        Config.__is_init_device = True
        logger.info(f"[{self.run_name}] Initialized device.")

    # ------------------------------
    # other functions
    # ------------------------------
    def to_dict(self) -> dict:
        conf = {}
        for k, v in self.__dict__.items():
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
        env_config = self.env_config.copy()
        rl_config = self.rl_config.copy()
        config = Config(env_config, rl_config)
        config._now = self._now

        # parameter
        for k, v in self.__dict__.items():
            if k == "_callbacks":
                continue
            if (v is None) or (type(v) in [int, float, bool, str]) or issubclass(type(v), enum.Enum):
                setattr(config, k, v)
            elif type(v) is list:
                arr = []
                for a in v:
                    if a is None:
                        arr.append(None)
                    else:
                        arr.append(pickle.loads(pickle.dumps(a)))
                setattr(config, k, arr)

        # callback
        if callbacks_share:
            config._callbacks = self.callbacks
        else:
            config._callbacks = pickle.loads(pickle.dumps(self.callbacks))

        # env
        if env_share:
            config.env = self.env

        return config

    # ------------------------------
    # utility
    # ------------------------------
    def get_env_init_state(self, encode: bool = True) -> Union[EnvObservationType, RLObservationType]:
        env = self.make_env()
        env.reset()
        state = env.state
        if encode:
            worker = self.make_worker()
            if isinstance(worker.worker, RLWorker):
                state = worker.worker.state_encode(state, env)
        return state

    def save(
        self,
        path: str,
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
    ) -> None:
        dat = [
            self,
            parameter.backup() if parameter is not None else None,
            remote_memory.backup(compress=True) if remote_memory is not None else None,
        ]
        with open(path, "wb") as f:
            pickle.dump(dat, f)

    @staticmethod
    def load(path: str) -> Tuple["Config", RLParameter, RLRemoteMemory]:
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
