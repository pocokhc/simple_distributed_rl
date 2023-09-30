import copy
import datetime
import logging
import os
import pickle
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np

from srl.base.define import EnvActionType, EnvObservationType, InfoType, PlayRenderModes, RLObservationType
from srl.base.env.config import EnvConfig
from srl.base.env.env_run import EnvRun
from srl.base.env.registration import make as make_env
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
from srl.runner.callback import CallbackType
from srl.utils import common
from srl.utils.serialize import convert_for_json

if TYPE_CHECKING:
    import psutil

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """
    実行前に設定される変数をまとめたクラス
    Class that summarizes variables to be set before execution
    """

    env_config: EnvConfig
    rl_config: RLConfig

    players: List[Union[None, str, Tuple[str, dict], RLConfig]] = field(default_factory=list)

    # --- mp
    trainer_parameter_send_interval_by_train_count: int = 100
    actor_parameter_sync_interval_by_step: int = 100
    # context
    actor_num: int = 1

    # --- remote
    remote_port: int = 50000
    remote_authkey: bytes = b"abracadabra"

    # --- random
    seed: Optional[int] = None
    seed_enable_gpu: bool = True  # 有効にならない場合あり、また速度が犠牲になる可能性あり

    # --- dir
    # 基本となるディレクトリ、ファイル関係は、この配下に時刻でフォルダが作られ、その下が基準となる
    base_dir: str = "tmp"

    # --- stats
    enable_stats: bool = True

    # --- device
    # option
    device_main: str = "AUTO"
    device_mp_trainer: str = "AUTO"
    device_mp_actors: Union[str, List[str]] = "AUTO"
    use_CUDA_VISIBLE_DEVICES: bool = True
    # tensorflow options
    tf_device_disable: bool = False
    tf_enable_memory_growth: bool = True

    def to_dict(self, mask: bool = True) -> dict:
        dat: dict = convert_for_json(self.__dict__)
        if mask:
            dat["remote_authkey"] = "mask"
        return dat

    def copy(self) -> "Config":
        return copy.deepcopy(self)


@dataclass
class Context:
    """
    実行直前に決定される変数をまとめたクラス
    Class that summarizes variables to be determined just before execution
    """

    # --- process context
    run_name: str = "main"
    distributed: bool = False
    actor_id: int = 0

    # --- stats
    used_psutil: bool = False
    used_nvidia: bool = False

    # --- device
    framework: str = ""
    used_device_tf: str = "/CPU"
    used_device_torch: str = "cpu"

    # --- play context
    # stop config
    max_episodes: int = 0
    timeout: int = 0
    max_steps: int = 0
    max_train_count: int = 0
    # play config
    train_only: bool = False
    shuffle_player: bool = True
    disable_trainer: bool = False
    # play info
    training: bool = False
    render_mode: PlayRenderModes = PlayRenderModes.none
    # other
    start_date: datetime.datetime = datetime.datetime(2000, 1, 1)
    save_dir: str = ""

    # --- callbacks
    callbacks: List[CallbackType] = field(default_factory=list)

    def __post_init__(self):
        self._is_init = False

    def init(self, runner: "Runner"):
        if self._is_init:
            return

        # --- check stop config ---
        if self.train_only and (not self.distributed):
            self.disable_trainer = False
            self.training = True
            assert self.max_train_count > 0 or self.timeout > 0, "Please specify 'max_train_count' or 'timeout'."
        else:
            assert (
                self.max_steps > 0 or self.max_episodes > 0 or self.timeout > 0 or self.max_train_count > 0
            ), "Please specify 'max_episodes', 'timeout' , 'max_steps' or 'max_train_count'."
        # -------------------------

        # --- 作成できるかcheck
        runner.make_parameter()
        runner.make_remote_memory()

        # init
        self.render_mode = PlayRenderModes.from_str(self.render_mode)

        # main のみ更新
        self.start_date = datetime.datetime.now()

        # "YYYYMMDD_HHMMSS_EnvName_RLName"
        dir_name = self.start_date.strftime("%Y%m%d_%H%M%S")
        dir_name += f"_{runner.config.env_config.name}_{runner.config.rl_config.getName()}"
        dir_name = re.sub(r'[\\/:?."<>\|]', "_", dir_name)
        self.save_dir = os.path.join(runner.config.base_dir, dir_name)

        self._is_init = True

    def to_dict(self) -> dict:
        dat: dict = convert_for_json(self.__dict__)
        return dat

    def copy(self) -> "Context":
        return copy.deepcopy(self)


@dataclass
class State:
    """
    実行中に変動する変数をまとめたクラス
    Class that summarizes variables that change during execution
    """

    env: Optional[EnvRun] = None
    workers: List[WorkerRun] = field(default_factory=list)
    trainer: Optional[RLTrainer] = None
    remote_memory: Optional[RLRemoteMemory] = None
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

    # trainer
    train_info: Optional[InfoType] = None

    # mp
    sync_actor: int = 0
    sync_trainer: int = 0

    # ----------
    # user
    # ----------
    user_data: dict = field(default_factory=dict)

    def set_user_val(self, key: str, val: Any):
        self.user_data[key] = val

    def get_user_val(self, key: str):
        return self.user_data[key]

    # ------------

    def to_dict(self) -> dict:
        dat: dict = convert_for_json(self.__dict__)
        return dat


@dataclass()
class Runner:
    """実行環境を提供"""

    #: EnvConfigを指定（文字列のみのIDでも可能）
    name_or_env_config: Union[str, EnvConfig]
    #: RLConfigを指定
    #: Noneの場合、dummyアルゴリズムが使われます
    rl_config: Optional[RLConfig] = None  # type: ignore

    # --- private(static class instance)
    # multiprocessing("spawn")ではプロセス毎に初期化される想定
    # pynvmlはプロセス毎に管理
    __is_init_process = False
    __framework = ""
    __used_device_tf = "/CPU"
    __used_device_torch = "cpu"
    __used_nvidia = False
    __used_psutil = False

    def __post_init__(self):
        if isinstance(self.name_or_env_config, str):
            env_config = EnvConfig(self.name_or_env_config)
        else:
            env_config: EnvConfig = self.name_or_env_config
        if self.rl_config is None:
            self.rl_config = dummy.Config()
        self.rl_config: RLConfig = self.rl_config

        # --------------------------------------------
        self.config = Config(env_config, self.rl_config)
        self.context = Context()
        self.state = State()

        self._env = None
        self._parameter = None
        self._remote_memory = None

        self._history = []
        self._history_viewer = None
        self._history_on_memory_callback = None
        self._history_on_file_callback = None
        self._checkpoint_callback = None
        self._psutil_process: Optional["psutil.Process"] = None

    @property
    def env_config(self) -> EnvConfig:
        return self.config.env_config

    @property
    def env(self) -> EnvRun:
        return self.make_env()

    @property
    def parameter(self) -> RLParameter:
        return self.make_parameter()

    @property
    def remote_memory(self) -> RLRemoteMemory:
        return self.make_remote_memory()

    def get_history(self):
        assert self.history_viewer is not None
        return self.history_viewer

    # ------------------------------
    # set config
    # ------------------------------
    def set_players(self, players: List[Union[None, str, Tuple[str, dict], RLConfig]] = []):
        """multi player option
        マルチプレイヤーゲームでのプレイヤーを指定します。

        None             : use rl_config worker
        str              : Registered RuleWorker
        Tuple[str, dict] : Registered RuleWorker(Pass kwargs argument)
        RLConfig         : use RLConfig worker

        Args:
            players: マルチプレイヤーゲームにおけるプレイヤーを表した配列

        """
        # playersという変数名だけど、役割はworkersの方が正しい
        self.config.players = players

    def set_save_dir(self, save_dir: str):
        """ディレクトリへの保存が必要な場合、そのディレクトリを指定します。
        ディレクトリへの保存は、historyやcheckpointがあります。

        Args:
            save_dir (str): 保存ディレクトリのパス。 Defaults to "tmp".
        """
        self.config.base_dir = save_dir

    def set_mp(
        self,
        trainer_parameter_send_interval_by_train_count: int = 100,
        actor_parameter_sync_interval_by_step: int = 100,
    ):
        """分散学習時の設定を指定します。

        Args:
            trainer_parameter_send_interval_by_train_count (int, optional): Trainerがパラメターを同期する間隔. Defaults to 100.
            actor_parameter_sync_interval_by_step (int, optional): Actorがパラメータを同期する間隔. Defaults to 100.
        """
        self.config.trainer_parameter_send_interval_by_train_count = trainer_parameter_send_interval_by_train_count
        self.config.actor_parameter_sync_interval_by_step = actor_parameter_sync_interval_by_step

    def set_remote(
        self,
        remote_server_ip: str = "127.0.0.1",
        remote_port: int = 50000,
        remote_authkey: bytes = b"abracadabra",
    ):
        """分散コンピューティングによる学習の設定を指定します。
        （こちらの機能はまだ開発中）

        "0.0.0.0" for external publication

        Args:
            remote_server_ip (str, optional): ホストのIPアドレス. Defaults to "127.0.0.1".
            remote_port (int, optional): ホストのポート. Defaults to 50000.
            remote_authkey (bytes, optional): 接続に使う認証キー. Defaults to b"abracadabra".
        """
        self.config.remote_server_ip = remote_server_ip
        self.config.remote_port = remote_port
        self.config.remote_authkey = remote_authkey

    def set_seed(
        self,
        seed: Optional[int] = None,
        seed_enable_gpu: bool = True,
    ):
        """set random seed.

        Args:
            seed (Optional[int], optional): random seed. Defaults to None.
            seed_enable_gpu (bool, optional): set GPU seed(実行速度が遅くなる場合があります). Defaults to True.
        """
        self.config.seed = seed
        self.config.seed_enable_gpu = seed_enable_gpu

    # ------------------------------
    # model summary
    # ------------------------------
    def model_summary(self, **kwargs) -> RLParameter:
        """modelの概要を返します。これは以下と同じです。

        >>> parameter = runner.make_parameter()
        >>> parameter.summary()

        Returns:
            RLParameter: RLParameter
        """
        parameter = self.make_parameter()
        parameter.summary(**kwargs)
        return parameter

    # ------------------------------
    # save/load
    # ------------------------------
    def save_parameter(self, path: str):
        """save parameter"""
        self.make_parameter().save(path)

    def load_parameter(self, path: str):
        """load parameter"""
        self.make_parameter().load(path)

    def save_remote_memory(self, path: str, compress: bool = True, **kwargs):
        """save remote memory

        Args:
            path (str): save path
            compress (bool, optional): 圧縮するかどうか。圧縮はlzma形式です. Defaults to True.
        """
        self.make_remote_memory().save(path, compress, **kwargs)

    def load_remote_memory(self, path: str, **kwargs):
        """load remote memory"""
        self.make_remote_memory().load(path, **kwargs)

    # ------------------------------
    # make functions
    # ------------------------------
    def make_env(self, is_init: bool = False) -> EnvRun:
        if self._env is None:
            self._env = make_env(self.env_config)
            logger.info(f"make env: {self._env.name}")
        if is_init:
            self._env.init()
        return self._env

    def make_parameter(self, is_load: bool = True) -> RLParameter:
        self._init_process()
        if self._parameter is None:
            if not self.rl_config.is_reset:
                self.rl_config.reset(self.make_env())
            self._parameter = make_parameter(self.rl_config, is_load=is_load)
            logger.info(f"make parameter: {self._parameter}")
        return self._parameter

    def make_remote_memory(self, is_load: bool = True) -> RLRemoteMemory:
        self._init_process()
        if self._remote_memory is None:
            if not self.rl_config.is_reset:
                self.rl_config.reset(self.make_env())
            self._remote_memory = make_remote_memory(self.rl_config, is_load=is_load)
            logger.info(f"make remote_memory: {self._remote_memory}")
        return self._remote_memory

    def make_trainer(
        self,
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
    ) -> RLTrainer:
        if parameter is None:
            parameter = self.make_parameter()
        if remote_memory is None:
            remote_memory = self.make_remote_memory()
        if not self.rl_config.is_reset:
            self.rl_config.reset(self.make_env())
        return make_trainer(self.rl_config, parameter, remote_memory)

    def make_worker(
        self,
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
    ) -> WorkerRun:
        if parameter is None:
            parameter = self.make_parameter()
        if remote_memory is None:
            remote_memory = self.make_remote_memory()
        if not self.rl_config.is_reset:
            self.rl_config.reset(self.make_env())
        return make_worker(
            self.rl_config,
            self.make_env(),
            parameter,
            remote_memory,
            self.context.distributed,
            self.context.actor_id,
        )

    def make_player(
        self,
        player: Union[None, str, RLConfig],
        worker_kwargs={},
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
    ) -> WorkerRun:
        env = self.make_env()

        # none はベース
        if player is None:
            return self.make_worker(parameter, remote_memory)

        # 文字列はenv側またはルールベースのアルゴリズム
        if isinstance(player, str):
            worker = env.make_worker(
                player,
                self.context.distributed,
                enable_raise=False,
                env_worker_kwargs=worker_kwargs,
            )
            if worker is not None:
                return worker
            worker = make_worker_rulebase(
                player,
                env,
                update_config_parameter=worker_kwargs,
                distributed=self.context.distributed,
                actor_id=self.context.actor_id,
                is_reset_logger=False,
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
                self.context.distributed,
                self.context.actor_id,
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
        if len(self.config.players) == 0:
            players: List[Union[None, str, Tuple[str, dict], RLConfig]] = ["random" for _ in range(env.player_num)]
            players[0] = None
        else:
            players = self.config.players

        workers = []
        for i in range(env.player_num):
            p = players[i] if i < len(players) else None
            kwargs = {}
            if isinstance(p, tuple) or isinstance(p, list):
                kwargs = p[1]
                p = p[0]
            workers.append(self.make_player(p, kwargs, parameter, remote_memory))
        return workers

    # ------------------------------
    # process
    # ------------------------------
    def _init_process(self):
        if Runner.__is_init_process:
            # 一度決定した値を使う
            self.context.framework = Runner.__framework
            self.context.used_device_tf = Runner.__used_device_tf
            self.context.used_device_torch = Runner.__used_device_torch
            self.rl_config._used_device_tf = self.context.used_device_tf
            self.rl_config._used_device_torch = self.context.used_device_torch
            self.context.used_nvidia = Runner.__used_nvidia
            self.context.used_psutil = Runner.__used_psutil
            # assert self.context.framework == framework, 別のframeworkを併用する場合の動作は未定義
            return

        self.__init_nvidia()
        self.__init_psutil()
        self.__init_device()

        Runner.__framework = self.context.framework
        Runner.__used_device_tf = self.context.used_device_tf
        Runner.__used_device_torch = self.context.used_device_torch
        Runner.__used_nvidia = self.context.used_nvidia
        Runner.__used_psutil = self.context.used_psutil
        Runner.__is_init_process = True

    # --- system profile
    def set_stats(self, enable_stats: bool = True):
        """ハードウェアの統計情報に関する設定を指定します。

        Args:
            enable_stats (bool, optional): 統計情報の取得をするかどうか. Defaults to True.
        """
        self.config.enable_stats = enable_stats

    def __init_nvidia(self):
        if not self.config.enable_stats:
            return

        self.context.used_nvidia = False
        if common.is_package_installed("pynvml"):
            try:
                import pynvml

                pynvml.nvmlInit()
                self.context.used_nvidia = True

            except Exception as e:
                import traceback

                logger.debug(traceback.format_exc())
                logger.info(e)

    def close_nvidia(self):
        if Runner.__used_nvidia:
            Runner.__used_nvidia = False
            self.context.used_nvidia = False
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:
                import traceback

                logger.info(traceback.format_exc())

    def read_nvml(self) -> List[Tuple[int, float, float]]:
        if self.context.used_nvidia:
            import pynvml

            gpu_num = pynvml.nvmlDeviceGetCount()
            gpus = []
            for device_id in range(gpu_num):
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpus.append((device_id, float(rate.gpu), float(rate.memory)))
            return gpus
        return []

    def __init_psutil(self):
        if not self.config.enable_stats:
            return

        self.context.used_psutil = False
        if common.is_package_installed("psutil"):
            try:
                import psutil

                self._psutil_process = psutil.Process()
                self.context.used_psutil = True
            except Exception as e:
                import traceback

                logger.debug(traceback.format_exc())
                logger.info(e)

    def read_psutil(self) -> Tuple[float, float]:
        if self._psutil_process is None:
            return np.NaN, np.NaN

        import psutil

        # CPU,memory
        memory_percent = psutil.virtual_memory().percent
        cpu_percent = self._psutil_process.cpu_percent(None) / psutil.cpu_count()

        return memory_percent, cpu_percent

    # --- device
    def set_device(
        self,
        device_main: str = "AUTO",
        device_mp_trainer: str = "AUTO",
        device_mp_actors: Union[str, List[str]] = "AUTO",  # ["CPU:0", "CPU:1"]
        use_CUDA_VISIBLE_DEVICES: bool = True,
        tf_device_disable: bool = False,
        tf_enable_memory_growth: bool = True,
    ):
        """set device.

        "AUTO",""    : Automatic assignment.
        "CPU","CPU:0": Use CPU.
        "GPU","GPU:0": Use GPU.

        AUTO assign
          sequence
            main   : GPU -> CPU
            trainer: not use
            actors : not use
          distribute
            main   : CPU
            trainer: GPU -> CPU
            actors : CPU

        Args:
            device_main (str, optional): mainのdeviceを指定します。分散学習を用いない場合、これだけが使用されます. Defaults to "AUTO".
            device_mp_trainer (str, optional): 分散学習時のtrainerが使うdeviceを指定します. Defaults to "AUTO".
            device_mp_actors (Union[str, List[str]], optional): 分散学習時のactorが使うdeviceを指定します. Defaults to "AUTO".
            use_CUDA_VISIBLE_DEVICES (bool, optional): CPUの場合 CUDA_VISIBLE_DEVICES を-1にする. Defaults to True.
            tf_device_disable (bool, optional): tensorflowにて、 'with tf.device()' を使用する. Defaults to True.
            tf_enable_memory_growth (bool, optional): tensorflowにて、'set_memory_growth(True)' を実行する. Defaults to True.
        """
        if Runner.__is_init_process:
            logger.warning("Device cannot be changed after initialization.")
            return

        self.config.device_main = device_main.upper()
        self.config.device_mp_trainer = device_mp_trainer.upper()
        if isinstance(device_mp_actors, str):
            device_mp_actors = device_mp_actors.upper()
        else:
            device_mp_actors = [d.upper() for d in device_mp_actors]
        self.config.device_mp_actors = device_mp_actors
        self.config.use_CUDA_VISIBLE_DEVICES = use_CUDA_VISIBLE_DEVICES
        self.config.tf_device_disable = tf_device_disable
        self.config.tf_enable_memory_growth = tf_enable_memory_growth

    def __init_device(self):
        # frameworkは "" の場合何もしない(フラグも立てない)
        framework = self.rl_config.get_use_framework()
        if framework == "":
            return
        if framework == "tf":
            framework = "tensorflow"
        assert framework in ["tensorflow", "torch"], "Framework can specify 'tensorflow' or 'torch'."
        self.context.framework = framework

        run_name = self.context.run_name
        distributed = self.context.distributed
        actor_id = self.context.actor_id

        # logger
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            cuda_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            logger.info(f"[{run_name}] CUDA_VISIBLE_DEVICES='{cuda_devices}'")
        else:
            logger.info(f"[{run_name}] CUDA_VISIBLE_DEVICES is not define.")

        # --- check device ---
        if run_name in ["main", "eval"]:
            device = self.config.device_main
            if device in ["", "AUTO"]:
                if distributed:
                    device = "CPU"
                else:
                    device = "AUTO"
        elif run_name == "trainer":
            device = self.config.device_mp_trainer
            if device == "":
                device = "AUTO"
        elif "actor" in run_name:
            if isinstance(self.config.device_mp_actors, str):
                device = self.config.device_mp_actors
            else:
                device = self.config.device_mp_actors[actor_id]
            if device in ["", "AUTO"]:
                device = "CPU"
        else:
            raise ValueError(f"not coming(run_name={run_name})")
        logger.info(f"[{run_name}] used device name: {device}")
        # -----------------------

        # --- CUDA_VISIBLE_DEVICES ---
        if self.config.use_CUDA_VISIBLE_DEVICES:
            if "CPU" in device:
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                logger.info(f"[{run_name}] set CUDA_VISIBLE_DEVICES=-1")
            else:
                # CUDA_VISIBLE_DEVICES が -1 の場合のみ削除する
                if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "-1":
                    del os.environ["CUDA_VISIBLE_DEVICES"]
                    logger.info(f"[{run_name}] del CUDA_VISIBLE_DEVICES")
        # -----------------------------

        # --- CPU ---
        if "CPU" in device:
            if "CPU" == device:
                self.context.used_device_tf = "/CPU"
                self.context.used_device_torch = "cpu"
            elif "CPU:" in device:
                t = device.split(":")
                self.context.used_device_tf = f"/CPU:{t[1]}"
                self.context.used_device_torch = f"cpu:{t[1]}"
        # -----------

        # --- tf memory growth ---
        # Tensorflow,GPU がある場合に実施(CPUにしてもなぜかGPUの初期化は走る場合あり)
        if framework == "tensorflow" and self.config.tf_enable_memory_growth:
            try:
                import tensorflow as tf

                gpu_devices = tf.config.list_physical_devices("GPU")
                for d in gpu_devices:
                    logger.info(f"[{run_name}] (tf) set_memory_growth({d.name}, True)")
                    tf.config.experimental.set_memory_growth(d, True)
            except Exception:
                s = f"[{run_name}] (tf) 'set_memory_growth' failed."
                s += " Also consider 'runner.set_device(tf_enable_memory_growth=False)'."
                print(s)
                raise
        # -----------------------

        # --- GPU ---
        if "CPU" not in device:  # AUTOの場合もあり
            # CUDA_VISIBLE_DEVICES が -1 の場合のみ削除する
            if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "-1":
                del os.environ["CUDA_VISIBLE_DEVICES"]
                logger.info(f"[{run_name}] del CUDA_VISIBLE_DEVICES")

            # --- tensorflow GPU check
            if framework == "tensorflow":
                import tensorflow as tf

                gpu_devices = tf.config.list_physical_devices("GPU")
                if len(gpu_devices) == 0:
                    assert (
                        "GPU" not in device
                    ), f"[{run_name}] (tf) GPU is not found. {tf.config.list_physical_devices()}"

                    self.context.used_device_tf = "/CPU"

                else:
                    logger.info(f"[{run_name}] (tf) gpu device: {len(gpu_devices)}")

                    if "GPU:" in device:
                        t = device.split(":")
                        self.context.used_device_tf = f"/GPU:{t[1]}"
                    else:
                        self.context.used_device_tf = "/GPU"

            # --- torch GPU check
            if framework == "torch":
                import torch

                if torch.cuda.is_available():
                    logger.info(f"[{run_name}] (torch) gpu device: {torch.cuda.get_device_name()}")

                    if "GPU:" in device:
                        t = device.split(":")
                        self.context.used_device_torch = f"cuda:{t[1]}"
                    else:
                        self.context.used_device_torch = "cuda"
                else:
                    assert "GPU" not in device, f"[{run_name}] (torch) GPU is not found."

                    self.context.used_device_torch = "cpu"
        # -------------------------

        self.rl_config._used_device_tf = self.context.used_device_tf
        self.rl_config._used_device_torch = self.context.used_device_torch
        logger.info(f"[{run_name}] Initialized device.")

    # ------------------------------
    # utility
    # ------------------------------
    def copy(self, copy_context: bool = False, env_share: bool = False) -> "Runner":
        runner = Runner(self.env_config, self.rl_config)
        runner.config = self.config.copy()
        if copy_context:
            runner.context = self.context.copy()

        if env_share:
            runner._env = self._env

        return runner

    def get_env_init_state(self, encode: bool = True) -> Union[EnvObservationType, RLObservationType]:
        env = self.make_env()
        env.reset()
        state = env.state
        if encode:
            worker = self.make_worker()
            state = worker.state_encode(state, env, append_recent_state=False)
        return state

    def save(self, path: str) -> None:
        dat = [
            self.config,
            self.context,
            self.parameter.backup(),
            self.remote_memory.backup(compress=True),
        ]
        with open(path, "wb") as f:
            pickle.dump(dat, f)

    @staticmethod
    def load(path: str) -> "Runner":
        with open(path, "rb") as f:
            dat = pickle.load(f)
        config = dat[0]
        runner = Runner(config.env_config, config.rl_config)
        runner.config = config
        runner.context = dat[1]
        runner.parameter.restore(dat[2])
        runner.remote_memory.restore(dat[3])
        return runner

    def create_eval_runner(self, env_share: bool = False) -> "Runner":
        eval_runner = self.copy(copy_context=False, env_share=env_share)
        # context
        c = Context()
        c.run_name = "eval"
        c.disable_trainer = True
        c.training = False
        eval_runner.context = c
        return eval_runner

    # ------------------------------
    # run
    # ------------------------------
    def _create_play_state(self) -> State:
        self.state = State()
        return self.state

    def _play(
        self,
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
    ):
        from .core import play

        if parameter is None:
            parameter = self.make_parameter()
        if remote_memory is None:
            remote_memory = self.make_remote_memory()
        play(self, parameter, remote_memory)

    def set_history(
        self,
        enable_history: bool = True,
        write_memory: bool = True,
        write_file: bool = False,
        interval: int = 1,
        enable_eval: bool = False,
        eval_env_sharing: bool = False,
        eval_episode: int = 1,
        eval_timeout: int = -1,
        eval_max_steps: int = -1,
        eval_players: List[Union[None, str, Tuple[str, dict], RLConfig]] = [],
        eval_shuffle_player: bool = False,
        eval_used_device_tf: str = "/CPU",
        eval_used_device_torch: str = "cpu",
        eval_callbacks: List[CallbackType] = [],
    ):
        """学習履歴を保存する設定を指定します。

        Args:
            enable_history (bool, optional): 学習履歴の保存を有効にします. Defaults to True.
            write_memory (bool, optional): 学習履歴をメモリに保存します。これは分散学習では無効になります. Defaults to True.
            write_file (bool, optional): 学習履歴をディスクに保存します。 Defaults to False.
            interval (int, optional): 学習履歴を保存する間隔(秒). Defaults to 1.
            enable_eval (bool, optional): 学習履歴の保存時に評価用のシミュレーションを実行します. Defaults to False.
            eval_env_sharing (bool, optional): 評価時に学習時のenvを共有します. Defaults to False.
            eval_episode (int, optional): 評価時のエピソード数. Defaults to 1.
            eval_timeout (int, optional): 評価時の1エピソードの制限時間. Defaults to -1.
            eval_max_steps (int, optional): 評価時の1エピソードの最大ステップ数. Defaults to -1.
            eval_players (List[Union[None, str, Tuple[str, dict], RLConfig]], optional): 評価時のplayers. Defaults to [].
            eval_shuffle_player (bool, optional): 評価時にplayersをシャッフルするか. Defaults to False.
            eval_used_device_tf (str, optional): 評価時のdevice. Defaults to "/CPU".
            eval_used_device_torch (str, optional): 評価時のdevice. Defaults to "cpu".
            eval_callbacks (List[CallbackType], optional): 評価時のcallbacks. Defaults to [].
        """
        self._history_on_memory_callback = None
        self._history_on_file_callback = None
        if enable_history:
            if write_memory:
                from srl.runner.callbacks.history_on_memory import HistoryOnMemory

                self._history_on_memory_callback = HistoryOnMemory(
                    interval=interval,
                    enable_eval=enable_eval,
                    eval_env_sharing=eval_env_sharing,
                    eval_episode=eval_episode,
                    eval_timeout=eval_timeout,
                    eval_max_steps=eval_max_steps,
                    eval_players=eval_players,
                    eval_shuffle_player=eval_shuffle_player,
                    eval_used_device_tf=eval_used_device_tf,
                    eval_used_device_torch=eval_used_device_torch,
                    eval_callbacks=eval_callbacks,
                )
            if write_file:
                from srl.runner.callbacks.history_on_file import HistoryOnFile

                self._history_on_file_callback = HistoryOnFile(
                    interval=interval,
                    enable_eval=enable_eval,
                    eval_env_sharing=eval_env_sharing,
                    eval_episode=eval_episode,
                    eval_timeout=eval_timeout,
                    eval_max_steps=eval_max_steps,
                    eval_players=eval_players,
                    eval_shuffle_player=eval_shuffle_player,
                    eval_used_device_tf=eval_used_device_tf,
                    eval_used_device_torch=eval_used_device_torch,
                    eval_callbacks=eval_callbacks,
                )

    def set_checkpoint(
        self,
        enable_checkpoint: bool = True,
        interval: int = 60 * 20,
        enable_eval: bool = True,
        eval_env_sharing: bool = False,
        eval_episode: int = 1,
        eval_timeout: int = -1,
        eval_max_steps: int = -1,
        eval_players: List[Union[None, str, Tuple[str, dict], RLConfig]] = [],
        eval_shuffle_player: bool = False,
        eval_used_device_tf: str = "/CPU",
        eval_used_device_torch: str = "cpu",
        eval_callbacks: List[CallbackType] = [],
    ):
        """一定間隔でモデルを保存します。

        Args:
            enable_checkpoint (bool, optional): checkpointを有効にします。 Defaults to True.
            interval (int, optional): 保存する間隔（秒）. Defaults to 60*20.
            enable_eval (bool, optional): モデル保存時に評価用のシミュレーションを実行します. Defaults to False.
            eval_env_sharing (bool, optional): 評価時に学習時のenvを共有します. Defaults to False.
            eval_episode (int, optional): 評価時のエピソード数. Defaults to 1.
            eval_timeout (int, optional): 評価時の1エピソードの制限時間. Defaults to -1.
            eval_max_steps (int, optional): 評価時の1エピソードの最大ステップ数. Defaults to -1.
            eval_players (List[Union[None, str, Tuple[str, dict], RLConfig]], optional): 評価時のplayers. Defaults to [].
            eval_shuffle_player (bool, optional): 評価時にplayersをシャッフルするか. Defaults to False.
            eval_used_device_tf (str, optional): 評価時のdevice. Defaults to "/CPU".
            eval_used_device_torch (str, optional): 評価時のdevice. Defaults to "cpu".
            eval_callbacks (List[CallbackType], optional): 評価時のcallbacks. Defaults to [].
        """
        if not enable_checkpoint:
            self._checkpoint_callback = None
            return

        from srl.runner.callbacks.checkpoint import Checkpoint

        self._checkpoint_callback = Checkpoint(
            interval=interval,
            enable_eval=enable_eval,
            eval_env_sharing=eval_env_sharing,
            eval_episode=eval_episode,
            eval_timeout=eval_timeout,
            eval_max_steps=eval_max_steps,
            eval_players=eval_players,
            eval_shuffle_player=eval_shuffle_player,
            eval_used_device_tf=eval_used_device_tf,
            eval_used_device_torch=eval_used_device_torch,
            eval_callbacks=eval_callbacks,
        )

    def train(
        self,
        # --- stop config
        max_episodes: int = -1,
        timeout: int = -1,
        max_steps: int = -1,
        max_train_count: int = -1,
        # --- play config
        shuffle_player: bool = True,
        disable_trainer: bool = False,
        # --- progress
        enable_progress: bool = True,
        progress_start_time: int = 1,
        progress_interval_limit: int = 60 * 10,
        progress_env_info: bool = False,
        progress_train_info: bool = True,
        progress_worker_info: bool = True,
        progress_worker: int = 0,
        # --- eval
        enable_eval: bool = False,
        eval_env_sharing: bool = False,
        eval_episode: int = 1,
        eval_timeout: int = -1,
        eval_max_steps: int = -1,
        eval_players: List[Union[None, str, Tuple[str, dict], RLConfig]] = [],
        eval_shuffle_player: bool = False,
        eval_used_device_tf: str = "/CPU",
        eval_used_device_torch: str = "cpu",
        eval_callbacks: List[CallbackType] = [],
        # --- other
        callbacks: List[CallbackType] = [],
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
    ):
        """train

        Args:
            max_episodes (int, optional): 終了するまでのエピソード数. Defaults to -1.
            timeout (int, optional): 終了するまでの時間（秒）. Defaults to -1.
            max_steps (int, optional): 終了するまでの総ステップ. Defaults to -1.
            max_train_count (int, optional): 終了するまでの学習回数. Defaults to -1.
            shuffle_player (bool, optional): playersをシャッフルするかどうか. Defaults to True.
            disable_trainer (bool, optional): Trainerを無効にするか。主に経験の実集めたい場合に使用。 Defaults to False.
            enable_progress (bool, optional): 進捗を表示するか. Defaults to True.
            progress_start_time (int, optional): 最初に進捗を表示する秒数. Defaults to 1.
            progress_interval_limit (int, optional): 進捗を表示する最大の間隔（秒）. Defaults to 60*10.
            progress_env_info (bool, optional): 進捗表示にenv infoを表示するか. Defaults to False.
            progress_train_info (bool, optional): 進捗表示にtrain infoを表示するか. Defaults to True.
            progress_worker_info (bool, optional): 進捗表示にworker infoを表示するか. Defaults to True.
            progress_worker (int, optional): 進捗表示に表示するworker index. Defaults to 0.
            enable_eval (bool, optional): 評価用のシミュレーションを実行します. Defaults to False.
            eval_env_sharing (bool, optional): 評価時に学習時のenvを共有します. Defaults to False.
            eval_episode (int, optional): 評価時のエピソード数. Defaults to 1.
            eval_timeout (int, optional): 評価時の1エピソードの制限時間. Defaults to -1.
            eval_max_steps (int, optional): 評価時の1エピソードの最大ステップ数. Defaults to -1.
            eval_players (List[Union[None, str, Tuple[str, dict], RLConfig]], optional): 評価時のplayers. Defaults to [].
            eval_shuffle_player (bool, optional): 評価時にplayersをシャッフルするか. Defaults to False.
            eval_used_device_tf (str, optional): 評価時のdevice. Defaults to "/CPU".
            eval_used_device_torch (str, optional): 評価時のdevice. Defaults to "cpu".
            eval_callbacks (List[CallbackType], optional): 評価時のcallbacks. Defaults to [].
            callbacks (List[CallbackType], optional): callbacks. Defaults to [].
            parameter (Optional[RLParameter], optional): 任意のParameter. Defaults to None.
            remote_memory (Optional[RLRemoteMemory], optional): 任意のRemoteMemory. Defaults to None.
        """

        self.context.callbacks = callbacks[:]

        # --- set context
        self.context.run_name = "main"
        self.context.distributed = False
        # stop config
        self.context.max_episodes = max_episodes
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = max_train_count
        # play config
        self.context.train_only = False
        self.context.shuffle_player = shuffle_player
        self.context.disable_trainer = disable_trainer
        # play info
        self.context.training = True
        self.context.render_mode = PlayRenderModes.none

        # init
        self.context.init(self)

        # --- progress ---
        if enable_progress:
            from srl.runner.callbacks.print_progress import PrintProgress

            self.context.callbacks.append(
                PrintProgress(
                    start_time=progress_start_time,
                    interval_limit=progress_interval_limit,
                    progress_env_info=progress_env_info,
                    progress_train_info=progress_train_info,
                    progress_worker_info=progress_worker_info,
                    progress_worker=progress_worker,
                    progress_max_actor=5,
                    enable_eval=enable_eval,
                    eval_env_sharing=eval_env_sharing,
                    eval_episode=eval_episode,
                    eval_timeout=eval_timeout,
                    eval_max_steps=eval_max_steps,
                    eval_players=eval_players,
                    eval_shuffle_player=eval_shuffle_player,
                    eval_used_device_tf=eval_used_device_tf,
                    eval_used_device_torch=eval_used_device_torch,
                    eval_callbacks=eval_callbacks,
                )
            )
        # ----------------

        # --- checkpoint ---
        if self._checkpoint_callback is not None:
            self.context.callbacks.append(self._checkpoint_callback)
        # ------------------

        # --- history ---
        if self._history_on_memory_callback is not None:
            self.context.callbacks.append(self._history_on_memory_callback)
        if self._history_on_file_callback is not None:
            self.context.callbacks.append(self._history_on_file_callback)
        # ----------------

        from .core import play

        if parameter is None:
            parameter = self.make_parameter()
        if remote_memory is None:
            remote_memory = self.make_remote_memory()
        play(self, parameter, remote_memory)

        # --- history ---
        if self._history_on_memory_callback is not None:
            from srl.runner.callbacks.history_viewer import HistoryViewer

            self.history_viewer = HistoryViewer()
            self.history_viewer.set_history_on_memory(self)
        elif self._history_on_file_callback is not None:
            from srl.runner.callbacks.history_viewer import HistoryViewer

            self.history_viewer = HistoryViewer()
            self.history_viewer.load(self.context.save_dir)
        # ----------------

    def train_only(
        self,
        # --- stop config
        timeout: int = -1,
        max_train_count: int = -1,
        # --- progress
        enable_progress: bool = True,
        progress_start_time: int = 1,
        progress_interval_limit: int = 60 * 10,
        progress_train_info: bool = True,
        # --- eval
        enable_eval: bool = False,
        eval_episode: int = 1,
        eval_timeout: int = -1,
        eval_max_steps: int = -1,
        eval_players: List[Union[None, str, Tuple[str, dict], RLConfig]] = [],
        eval_shuffle_player: bool = False,
        eval_used_device_tf: str = "/CPU",
        eval_used_device_torch: str = "cpu",
        eval_callbacks: List[CallbackType] = [],
        # --- other
        callbacks: List[CallbackType] = [],
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
    ):
        """Trainerが学習するだけでWorkerによるシミュレーションはありません。

        Args:
            max_episodes (int, optional): 終了するまでのエピソード数. Defaults to -1.
            max_train_count (int, optional): 終了するまでの学習回数. Defaults to -1.
            enable_progress (bool, optional): 進捗を表示するか. Defaults to True.
            progress_start_time (int, optional): 最初に進捗を表示する秒数. Defaults to 1.
            progress_interval_limit (int, optional): 進捗を表示する最大の間隔（秒）. Defaults to 60*10.
            progress_train_info (bool, optional): _description_. Defaults to True.
            enable_eval (bool, optional): 評価用のシミュレーションを実行します. Defaults to False.
            eval_episode (int, optional): 評価時のエピソード数. Defaults to 1.
            eval_timeout (int, optional): 評価時の1エピソードの制限時間. Defaults to -1.
            eval_max_steps (int, optional): 評価時の1エピソードの最大ステップ数. Defaults to -1.
            eval_players (List[Union[None, str, Tuple[str, dict], RLConfig]], optional): 評価時のplayers. Defaults to [].
            eval_shuffle_player (bool, optional): 評価時にplayersをシャッフルするか. Defaults to False.
            eval_used_device_tf (str, optional): 評価時のdevice. Defaults to "/CPU".
            eval_used_device_torch (str, optional): 評価時のdevice. Defaults to "cpu".
            eval_callbacks (List[CallbackType], optional): 評価時のcallbacks. Defaults to [].
            callbacks (List[CallbackType], optional): callbacks. Defaults to [].
            parameter (Optional[RLParameter], optional): 任意のParameter. Defaults to None.
            remote_memory (Optional[RLRemoteMemory], optional): 任意のRemoteMemory. Defaults to None.
        """
        self.context.callbacks = callbacks[:]

        # --- set context
        self.context.run_name = "main"
        self.context.distributed = False
        # stop config
        self.context.timeout = timeout
        self.context.max_train_count = max_train_count
        # play config
        self.context.train_only = True
        self.context.disable_trainer = False
        # play info
        self.context.training = True
        self.context.render_mode = PlayRenderModes.none

        # init
        self.context.init(self)

        # --- progress ---
        if enable_progress:
            from srl.runner.callbacks.print_progress import PrintProgress

            self.context.callbacks.append(
                PrintProgress(
                    start_time=progress_start_time,
                    interval_limit=progress_interval_limit,
                    progress_train_info=progress_train_info,
                    progress_max_actor=5,
                    enable_eval=enable_eval,
                    eval_env_sharing=True,
                    eval_episode=eval_episode,
                    eval_timeout=eval_timeout,
                    eval_max_steps=eval_max_steps,
                    eval_players=eval_players,
                    eval_shuffle_player=eval_shuffle_player,
                    eval_used_device_tf=eval_used_device_tf,
                    eval_used_device_torch=eval_used_device_torch,
                    eval_callbacks=eval_callbacks,
                )
            )
        # ----------------

        # --- checkpoint ---
        if self._checkpoint_callback is not None:
            self.context.callbacks.append(self._checkpoint_callback)
        # ------------------

        # --- history ---
        if self._history_on_memory_callback is not None:
            self.context.callbacks.append(self._history_on_memory_callback)
        if self._history_on_file_callback is not None:
            self.context.callbacks.append(self._history_on_file_callback)
        # ----------------

        from .core import play

        if parameter is None:
            parameter = self.make_parameter()
        if remote_memory is None:
            remote_memory = self.make_remote_memory()
        play(self, parameter, remote_memory)

        # --- history ---
        if self._history_on_memory_callback is not None:
            from srl.runner.callbacks.history_viewer import HistoryViewer

            self.history_viewer = HistoryViewer()
            self.history_viewer.set_history_on_memory(self)
        elif self._history_on_file_callback is not None:
            from srl.runner.callbacks.history_viewer import HistoryViewer

            self.history_viewer = HistoryViewer()
            self.history_viewer.load(self.context.save_dir)
        # ----------------

    def train_mp(
        self,
        # mp
        actor_num: int = 1,
        # --- stop config
        max_episodes: int = -1,
        timeout: int = -1,
        max_steps: int = -1,
        max_train_count: int = -1,
        # --- play config
        shuffle_player: bool = True,
        disable_trainer: bool = False,
        # --- progress
        enable_progress: bool = True,
        progress_start_time: int = 1,
        progress_interval_limit: int = 60 * 10,
        progress_env_info: bool = False,
        progress_train_info: bool = True,
        progress_worker_info: bool = True,
        progress_worker: int = 0,
        progress_max_actor: int = 5,
        # --- eval
        enable_eval: bool = False,
        eval_env_sharing: bool = False,
        eval_episode: int = 1,
        eval_timeout: int = -1,
        eval_max_steps: int = -1,
        eval_players: List[Union[None, str, Tuple[str, dict], RLConfig]] = [],
        eval_shuffle_player: bool = False,
        eval_used_device_tf: str = "/CPU",
        eval_used_device_torch: str = "cpu",
        eval_callbacks: List[CallbackType] = [],
        # --- other
        callbacks: List[CallbackType] = [],
        save_remote_memory: str = "",
        return_remote_memory: bool = False,
    ):
        """multiprocessingを使用した分散学習による学習を実施します。

        Args:
            actor_num (int, optional): actor数を指定. Defaults to 1.
            max_episodes (int, optional): 終了するまでのエピソード数. Defaults to -1.
            timeout (int, optional): 終了するまでの時間（秒）. Defaults to -1.
            max_steps (int, optional): 終了するまでの総ステップ. Defaults to -1.
            max_train_count (int, optional): 終了するまでの学習回数. Defaults to -1.
            shuffle_player (bool, optional): playersをシャッフルするかどうか. Defaults to True.
            disable_trainer (bool, optional): Trainerを無効にするか。主に経験の実集めたい場合に使用。 Defaults to False.
            enable_progress (bool, optional): 進捗を表示するか. Defaults to True.
            progress_start_time (int, optional): 最初に進捗を表示する秒数. Defaults to 1.
            progress_interval_limit (int, optional): 進捗を表示する最大の間隔（秒）. Defaults to 60*10.
            progress_env_info (bool, optional): 進捗表示にenv infoを表示するか. Defaults to False.
            progress_train_info (bool, optional): 進捗表示にtrain infoを表示するか. Defaults to True.
            progress_worker_info (bool, optional): 進捗表示にworker infoを表示するか. Defaults to True.
            progress_worker (int, optional): 進捗表示に表示するworker index. Defaults to 0.
            progress_max_actor (int, optional): 表示する最大actor数. Defaults to 5.
            enable_eval (bool, optional): 評価用のシミュレーションを実行します. Defaults to False.
            eval_env_sharing (bool, optional): 評価時に学習時のenvを共有します. Defaults to False.
            eval_episode (int, optional): 評価時のエピソード数. Defaults to 1.
            eval_timeout (int, optional): 評価時の1エピソードの制限時間. Defaults to -1.
            eval_max_steps (int, optional): 評価時の1エピソードの最大ステップ数. Defaults to -1.
            eval_players (List[Union[None, str, Tuple[str, dict], RLConfig]], optional): 評価時のplayers. Defaults to [].
            eval_shuffle_player (bool, optional): 評価時にplayersをシャッフルするか. Defaults to False.
            eval_used_device_tf (str, optional): 評価時のdevice. Defaults to "/CPU".
            eval_used_device_torch (str, optional): 評価時のdevice. Defaults to "cpu".
            eval_callbacks (List[CallbackType], optional): 評価時のcallbacks. Defaults to [].
            callbacks (List[CallbackType], optional): callbacks. Defaults to [].
            parameter (Optional[RLParameter], optional): 任意のParameter. Defaults to None.
            remote_memory (Optional[RLRemoteMemory], optional): 任意のRemoteMemory. Defaults to None.
        """
        self.context.callbacks = callbacks[:]
        self.config.actor_num = actor_num

        # --- set context
        self.context.run_name = "main"
        self.context.distributed = True
        # stop config
        self.context.max_episodes = max_episodes
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = max_train_count
        # play config
        self.context.train_only = False
        self.context.shuffle_player = shuffle_player
        self.context.disable_trainer = disable_trainer
        # play info
        self.context.training = True
        self.context.render_mode = PlayRenderModes.none

        # init
        self.context.init(self)

        # --- progress ---
        if enable_progress:
            from srl.runner.callbacks.print_progress import PrintProgress

            self.context.callbacks.append(
                PrintProgress(
                    start_time=progress_start_time,
                    interval_limit=progress_interval_limit,
                    progress_env_info=progress_env_info,
                    progress_train_info=progress_train_info,
                    progress_worker_info=progress_worker_info,
                    progress_worker=progress_worker,
                    progress_max_actor=progress_max_actor,
                    enable_eval=enable_eval,
                    eval_env_sharing=eval_env_sharing,
                    eval_episode=eval_episode,
                    eval_timeout=eval_timeout,
                    eval_max_steps=eval_max_steps,
                    eval_players=eval_players,
                    eval_shuffle_player=eval_shuffle_player,
                    eval_used_device_tf=eval_used_device_tf,
                    eval_used_device_torch=eval_used_device_torch,
                    eval_callbacks=eval_callbacks,
                )
            )
        # ----------------

        # --- checkpoint ---
        if self._checkpoint_callback is not None:
            self.context.callbacks.append(self._checkpoint_callback)
        # ------------------

        # --- history ---
        if self._history_on_file_callback is not None:
            self.context.callbacks.append(self._history_on_file_callback)
        # ----------------

        from .core_mp import train

        train(self, save_remote_memory, return_remote_memory)

        # --- history ---
        if self._history_on_file_callback is not None:
            from srl.runner.callbacks.history_viewer import HistoryViewer

            self.history_viewer = HistoryViewer()
            self.history_viewer.load(self.context.save_dir)
        # ----------------

    def train_mp_debug(
        self,
        # mp
        actor_num: int = 1,
        # --- stop config
        max_episodes: int = -1,
        timeout: int = -1,
        max_steps: int = -1,
        max_train_count: int = -1,
        # --- play config
        shuffle_player: bool = True,
        disable_trainer: bool = False,
        # --- progress
        enable_progress: bool = True,
        progress_start_time: int = 1,
        progress_interval_limit: int = 60 * 10,
        progress_env_info: bool = False,
        progress_train_info: bool = True,
        progress_worker_info: bool = True,
        progress_worker: int = 0,
        progress_max_actor: int = 5,
        # --- eval
        enable_eval: bool = False,
        eval_env_sharing: bool = False,
        eval_episode: int = 1,
        eval_timeout: int = -1,
        eval_max_steps: int = -1,
        eval_players: List[Union[None, str, Tuple[str, dict], RLConfig]] = [],
        eval_shuffle_player: bool = False,
        eval_used_device_tf: str = "/CPU",
        eval_used_device_torch: str = "cpu",
        eval_callbacks: List[CallbackType] = [],
        # --- other
        callbacks: List[CallbackType] = [],
        save_remote_memory: str = "",
        return_remote_memory: bool = False,
        # --- debug option
        choice_method: str = "random",
    ):
        """multiprocessingの分散学習と出来る限り似た学習を、single processで実施します。
        ほとんどの引数は train_mp と同じなのです。

        Args:
            choice_method(str, optional): 各actorとtrainerの採用方法を指定します. Defaults to 'random'.
        """
        self.context.callbacks = callbacks[:]
        self.config.actor_num = actor_num

        # --- set context
        self.context.run_name = "main"
        self.context.distributed = True
        # stop config
        self.context.max_episodes = max_episodes
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = max_train_count
        # play config
        self.context.train_only = False
        self.context.shuffle_player = shuffle_player
        self.context.disable_trainer = disable_trainer
        # play info
        self.context.training = True
        self.context.render_mode = PlayRenderModes.none

        # init
        self.context.init(self)

        # --- progress ---
        if enable_progress:
            from srl.runner.callbacks.print_progress import PrintProgress

            self.context.callbacks.append(
                PrintProgress(
                    start_time=progress_start_time,
                    interval_limit=progress_interval_limit,
                    progress_env_info=progress_env_info,
                    progress_train_info=progress_train_info,
                    progress_worker_info=progress_worker_info,
                    progress_worker=progress_worker,
                    progress_max_actor=progress_max_actor,
                    enable_eval=enable_eval,
                    eval_env_sharing=eval_env_sharing,
                    eval_episode=eval_episode,
                    eval_timeout=eval_timeout,
                    eval_max_steps=eval_max_steps,
                    eval_players=eval_players,
                    eval_shuffle_player=eval_shuffle_player,
                    eval_used_device_tf=eval_used_device_tf,
                    eval_used_device_torch=eval_used_device_torch,
                    eval_callbacks=eval_callbacks,
                )
            )
        # ----------------

        # --- checkpoint ---
        if self._checkpoint_callback is not None:
            self.context.callbacks.append(self._checkpoint_callback)
        # ------------------

        # --- history ---
        if self._history_on_file_callback is not None:
            self.context.callbacks.append(self._history_on_file_callback)
        # ----------------

        from .core_mp_debug import train

        train(self, save_remote_memory, return_remote_memory, choice_method)

        # --- history ---
        if self._history_on_file_callback is not None:
            from srl.runner.callbacks.history_viewer import HistoryViewer

            self.history_viewer = HistoryViewer()
            self.history_viewer.load(self.context.save_dir)
        # ----------------

    def train_remote(
        self,
        actor_num: int = 1,
        # --- stop config
        max_episodes: int = -1,
        timeout: int = -1,
        max_steps: int = -1,
        max_train_count: int = -1,
        # --- play config
        shuffle_player: bool = True,
        disable_trainer: bool = False,
        # --- progress
        enable_progress: bool = True,
        progress_start_time: int = 1,
        progress_interval_limit: int = 60 * 10,
        progress_env_info: bool = False,
        progress_train_info: bool = True,
        progress_worker_info: bool = True,
        progress_worker: int = 0,
        progress_max_actor: int = 5,
        # --- eval
        enable_eval: bool = False,
        eval_env_sharing: bool = False,
        eval_episode: int = 1,
        eval_timeout: int = -1,
        eval_max_steps: int = -1,
        eval_players: List[Union[None, str, Tuple[str, dict], RLConfig]] = [],
        eval_shuffle_player: bool = False,
        eval_used_device_tf: str = "/CPU",
        eval_used_device_torch: str = "cpu",
        eval_callbacks: List[CallbackType] = [],
        # --- other
        callbacks: List[CallbackType] = [],
    ):
        """
        Args:
            max_episodes (int, optional): 終了するまでのエピソード数. Defaults to -1.
            timeout (int, optional): 終了するまでの時間（秒）. Defaults to -1.
            max_steps (int, optional): 終了するまでの総ステップ. Defaults to -1.
            max_train_count (int, optional): 終了するまでの学習回数. Defaults to -1.
            shuffle_player (bool, optional): playersをシャッフルするかどうか. Defaults to True.
            disable_trainer (bool, optional): Trainerを無効にするか。主に経験の実集めたい場合に使用。 Defaults to False.
            enable_progress (bool, optional): 進捗を表示するか. Defaults to True.
            progress_start_time (int, optional): 最初に進捗を表示する秒数. Defaults to 1.
            progress_interval_limit (int, optional): 進捗を表示する最大の間隔（秒）. Defaults to 60*10.
            progress_env_info (bool, optional): 進捗表示にenv infoを表示するか. Defaults to False.
            progress_train_info (bool, optional): 進捗表示にtrain infoを表示するか. Defaults to True.
            progress_worker_info (bool, optional): 進捗表示にworker infoを表示するか. Defaults to True.
            progress_worker (int, optional): 進捗表示に表示するworker index. Defaults to 0.
            progress_max_actor (int, optional): 表示する最大actor数. Defaults to 5.
            enable_eval (bool, optional): 評価用のシミュレーションを実行します. Defaults to False.
            eval_env_sharing (bool, optional): 評価時に学習時のenvを共有します. Defaults to False.
            eval_episode (int, optional): 評価時のエピソード数. Defaults to 1.
            eval_timeout (int, optional): 評価時の1エピソードの制限時間. Defaults to -1.
            eval_max_steps (int, optional): 評価時の1エピソードの最大ステップ数. Defaults to -1.
            eval_players (List[Union[None, str, Tuple[str, dict], RLConfig]], optional): 評価時のplayers. Defaults to [].
            eval_shuffle_player (bool, optional): 評価時にplayersをシャッフルするか. Defaults to False.
            eval_used_device_tf (str, optional): 評価時のdevice. Defaults to "/CPU".
            eval_used_device_torch (str, optional): 評価時のdevice. Defaults to "cpu".
            eval_callbacks (List[CallbackType], optional): 評価時のcallbacks. Defaults to [].
            callbacks (List[CallbackType], optional): callbacks. Defaults to [].
        """
        self.context.callbacks = callbacks[:]
        self.config.actor_num = actor_num

        # --- set context
        self.context.run_name = "main"
        self.context.distributed = True
        # stop config
        self.context.max_episodes = max_episodes
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = max_train_count
        # play config
        self.context.train_only = False
        self.context.shuffle_player = shuffle_player
        self.context.disable_trainer = disable_trainer
        # play info
        self.context.training = True
        self.context.render_mode = PlayRenderModes.none

        # init
        self.context.init(self)

        # --- progress ---
        if enable_progress:
            from srl.runner.callbacks.print_progress import PrintProgress

            self.context.callbacks.append(
                PrintProgress(
                    start_time=progress_start_time,
                    interval_limit=progress_interval_limit,
                    progress_env_info=progress_env_info,
                    progress_train_info=progress_train_info,
                    progress_worker_info=progress_worker_info,
                    progress_worker=progress_worker,
                    progress_max_actor=progress_max_actor,
                    enable_eval=enable_eval,
                    eval_env_sharing=eval_env_sharing,
                    eval_episode=eval_episode,
                    eval_timeout=eval_timeout,
                    eval_max_steps=eval_max_steps,
                    eval_players=eval_players,
                    eval_shuffle_player=eval_shuffle_player,
                    eval_used_device_tf=eval_used_device_tf,
                    eval_used_device_torch=eval_used_device_torch,
                    eval_callbacks=eval_callbacks,
                )
            )
        # ----------------

        # --- checkpoint ---
        if self._checkpoint_callback is not None:
            self.context.callbacks.append(self._checkpoint_callback)
        # ------------------

        # --- history ---
        if self._history_on_file_callback is not None:
            self.context.callbacks.append(self._history_on_file_callback)
        # ----------------

        from .core_remote import train

        train(self)

        # --- history ---
        if self._history_on_file_callback is not None:
            from srl.runner.callbacks.history_viewer import HistoryViewer

            self.history_viewer = HistoryViewer()
            self.history_viewer.load(self.context.save_dir)
        # ----------------

    @staticmethod
    def run_actor(
        server_ip: str,
        port: int,
        authkey: bytes = b"abracadabra",
        actor_id: Optional[int] = None,
        verbose: bool = True,
    ):
        from .core_remote import run_actor

        run_actor(server_ip, port, authkey, actor_id, verbose)

    def evaluate(
        self,
        # --- stop config
        max_episodes: int = 10,
        timeout: int = -1,
        max_steps: int = -1,
        # --- play config
        shuffle_player: bool = True,
        # --- progress
        enable_progress: bool = True,
        progress_start_time: int = 1,
        progress_interval_limit: int = 60 * 10,
        progress_env_info: bool = False,
        progress_worker_info: bool = True,
        progress_worker: int = 0,
        # --- other
        callbacks: List[CallbackType] = [],
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
    ) -> Union[List[float], List[List[float]]]:  # single play , multi play
        """シミュレーションし、報酬を返します。

        Args:
            max_episodes (int, optional): 終了するまでのエピソード数. Defaults to 10.
            timeout (int, optional): 終了するまでの時間（秒）. Defaults to -1.
            max_steps (int, optional): 終了するまでの総ステップ. Defaults to -1.
            shuffle_player (bool, optional): playersをシャッフルするかどうか. Defaults to True.
            enable_progress (bool, optional): 進捗を表示するか. Defaults to True.
            progress_start_time (int, optional):  最初に進捗を表示する秒数. Defaults to 1.
            progress_interval_limit (int, optional): 進捗を表示する最大の間隔（秒）. Defaults to 60*10.
            progress_env_info (bool, optional): 進捗表示にenv infoを表示するか. Defaults to False.
            progress_worker_info (bool, optional): 進捗表示にworker infoを表示するか. Defaults to True.
            progress_worker (int, optional): 進捗表示に表示するworker index. Defaults to 0.
            callbacks (List[CallbackType], optional): callbacks. Defaults to [].
            parameter (Optional[RLParameter], optional): 任意のParameter. Defaults to None.
            remote_memory (Optional[RLRemoteMemory], optional): 任意のRemoteMemory. Defaults to None.

        Returns:
            Union[List[float], List[List[float]]]: プレイヤー数が1人なら Lost[float]、複数なら List[List[float]]] を返します。
        """
        self.context.callbacks = callbacks[:]

        # --- set context
        self.context.run_name = "main"
        self.context.distributed = False
        # stop config
        self.context.max_episodes = max_episodes
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        # play config
        self.context.train_only = False
        self.context.shuffle_player = shuffle_player
        self.context.disable_trainer = False
        # play info
        self.context.training = False
        self.context.render_mode = PlayRenderModes.none

        # init
        self.context.init(self)

        # --- progress ---
        if enable_progress:
            from srl.runner.callbacks.print_progress import PrintProgress

            self.context.callbacks.append(
                PrintProgress(
                    start_time=progress_start_time,
                    interval_limit=progress_interval_limit,
                    progress_env_info=progress_env_info,
                    progress_train_info=False,
                    progress_worker_info=progress_worker_info,
                    progress_worker=progress_worker,
                    progress_max_actor=5,
                    enable_eval=False,
                )
            )
        # ----------------

        from .core import play

        if parameter is None:
            parameter = self.make_parameter()
        if remote_memory is None:
            remote_memory = self.make_remote_memory()
        play(self, parameter, remote_memory)

        if self.env_config.player_num == 1:
            return [r[0] for r in self.state.episode_rewards_list]
        else:
            return self.state.episode_rewards_list

    def render_terminal(
        self,
        # rendering
        render_kwargs: dict = {},
        step_stop: bool = False,
        render_skip_step: bool = True,
        # --- stop config
        timeout: int = -1,
        max_steps: int = -1,
        # --- other
        callbacks: List[CallbackType] = [],
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
    ):
        """TODO"""
        self.context.callbacks = callbacks[:]

        mode = PlayRenderModes.terminal

        # --- set context
        self.context.run_name = "main"
        self.context.distributed = False
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        # play config
        self.context.train_only = False
        self.context.shuffle_player = False
        # play info
        self.context.training = False
        self.context.render_mode = mode

        # init
        self.context.init(self)

        # --- rendering ---
        from srl.runner.callbacks.rendering import Rendering

        self.context.callbacks.append(
            Rendering(
                mode=mode,
                kwargs=render_kwargs,
                step_stop=step_stop,
                render_skip_step=render_skip_step,
            )
        )
        # -----------------

        from .core import play

        if parameter is None:
            parameter = self.make_parameter()
        if remote_memory is None:
            remote_memory = self.make_remote_memory()
        play(self, parameter, remote_memory)

        return self.state.episode_rewards_list[0]

    def render_window(
        self,
        # rendering
        render_kwargs: dict = {},
        step_stop: bool = False,
        render_skip_step: bool = True,
        # render option
        render_interval: float = -1,  # ms
        render_scale: float = 1.0,
        font_name: str = "",
        font_size: int = 12,
        # --- stop config
        timeout: int = -1,
        max_steps: int = -1,
        # --- progress
        enable_progress: bool = True,
        progress_start_time: int = 1,
        progress_interval_limit: int = 60 * 10,
        progress_env_info: bool = False,
        progress_worker_info: bool = True,
        progress_worker: int = 0,
        # --- other
        callbacks: List[CallbackType] = [],
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
    ):
        """TODO"""

        self.context.callbacks = callbacks[:]

        mode = PlayRenderModes.window

        # --- set context
        self.context.run_name = "main"
        self.context.distributed = False
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        # play config
        self.context.train_only = False
        self.context.shuffle_player = False
        # play info
        self.context.training = False
        self.context.render_mode = mode

        # init
        self.context.init(self)

        # --- rendering ---
        from srl.runner.callbacks.rendering import Rendering

        self.context.callbacks.append(
            Rendering(
                mode=mode,
                kwargs=render_kwargs,
                step_stop=step_stop,
                render_skip_step=render_skip_step,
                render_interval=render_interval,
                render_scale=render_scale,
                font_name=font_name,
                font_size=font_size,
            )
        )
        # -----------------

        # --- progress ---
        if enable_progress:
            from srl.runner.callbacks.print_progress import PrintProgress

            self.context.callbacks.append(
                PrintProgress(
                    start_time=progress_start_time,
                    interval_limit=progress_interval_limit,
                    progress_env_info=progress_env_info,
                    progress_train_info=False,
                    progress_worker_info=progress_worker_info,
                    progress_worker=progress_worker,
                    progress_max_actor=5,
                    enable_eval=False,
                )
            )
        # ----------------

        from .core import play

        if parameter is None:
            parameter = self.make_parameter()
        if remote_memory is None:
            remote_memory = self.make_remote_memory()
        play(self, parameter, remote_memory)

        return self.state.episode_rewards_list[0]

    def animation_save_gif(
        self,
        path: str,
        # rendering
        render_kwargs: dict = {},
        step_stop: bool = False,
        render_skip_step: bool = True,
        # render option
        render_interval: float = -1,  # ms
        render_scale: float = 1.0,
        font_name: str = "",
        font_size: int = 12,
        #
        draw_info: bool = True,
        # --- stop config
        timeout: int = -1,
        max_steps: int = -1,
        # --- progress
        enable_progress: bool = True,
        progress_start_time: int = 1,
        progress_interval_limit: int = 60 * 10,
        progress_env_info: bool = False,
        progress_worker_info: bool = True,
        progress_worker: int = 0,
        # --- other
        callbacks: List[CallbackType] = [],
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
    ):
        """TODO"""

        self.context.callbacks = callbacks[:]

        mode = PlayRenderModes.rgb_array

        # --- set context
        self.context.run_name = "main"
        self.context.distributed = False
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        # play config
        self.context.train_only = False
        self.context.shuffle_player = False
        # play info
        self.context.training = False
        self.context.render_mode = mode

        # init
        self.context.init(self)

        # --- rendering ---
        from srl.runner.callbacks.rendering import Rendering

        rendering = Rendering(
            mode=mode,
            kwargs=render_kwargs,
            step_stop=step_stop,
            render_skip_step=render_skip_step,
            render_interval=-1,
            render_scale=1,
            font_name=font_name,
            font_size=font_size,
        )
        self.context.callbacks.append(rendering)
        # -----------------

        # --- progress ---
        if enable_progress:
            from srl.runner.callbacks.print_progress import PrintProgress

            self.context.callbacks.append(
                PrintProgress(
                    start_time=progress_start_time,
                    interval_limit=progress_interval_limit,
                    progress_env_info=progress_env_info,
                    progress_train_info=False,
                    progress_worker_info=progress_worker_info,
                    progress_worker=progress_worker,
                    progress_max_actor=5,
                    enable_eval=False,
                )
            )
        # ----------------

        from .core import play

        if parameter is None:
            parameter = self.make_parameter()
        if remote_memory is None:
            remote_memory = self.make_remote_memory()
        play(self, parameter, remote_memory)

        rendering.create_anime(render_interval, render_scale, draw_info).save(path)

        return self.state.episode_rewards_list[0]

    def animation_display(
        self,
        # rendering
        render_kwargs: dict = {},
        step_stop: bool = False,
        render_skip_step: bool = True,
        # render option
        render_interval: float = -1,  # ms
        render_scale: float = 1.0,
        font_name: str = "",
        font_size: int = 12,
        #
        draw_info: bool = True,
        # --- stop config
        timeout: int = -1,
        max_steps: int = -1,
        # --- progress
        enable_progress: bool = True,
        progress_start_time: int = 1,
        progress_interval_limit: int = 60 * 10,
        progress_env_info: bool = False,
        progress_worker_info: bool = True,
        progress_worker: int = 0,
        # --- other
        callbacks: List[CallbackType] = [],
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
    ):
        """TODO"""

        self.context.callbacks = callbacks[:]

        mode = PlayRenderModes.rgb_array

        # --- set context
        self.context.run_name = "main"
        self.context.distributed = False
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        # play config
        self.context.train_only = False
        self.context.shuffle_player = False
        # play info
        self.context.training = False
        self.context.render_mode = mode

        # init
        self.context.init(self)

        # --- rendering ---
        from srl.runner.callbacks.rendering import Rendering

        rendering = Rendering(
            mode=mode,
            kwargs=render_kwargs,
            step_stop=step_stop,
            render_skip_step=render_skip_step,
            render_interval=1,
            render_scale=1,
            font_name=font_name,
            font_size=font_size,
        )
        self.context.callbacks.append(rendering)
        # -----------------

        # --- progress ---
        if enable_progress:
            from srl.runner.callbacks.print_progress import PrintProgress

            self.context.callbacks.append(
                PrintProgress(
                    start_time=progress_start_time,
                    interval_limit=progress_interval_limit,
                    progress_env_info=progress_env_info,
                    progress_train_info=False,
                    progress_worker_info=progress_worker_info,
                    progress_worker=progress_worker,
                    progress_max_actor=5,
                    enable_eval=False,
                )
            )
        # ----------------

        from .core import play

        if parameter is None:
            parameter = self.make_parameter()
        if remote_memory is None:
            remote_memory = self.make_remote_memory()
        play(self, parameter, remote_memory)

        rendering.display(render_interval, render_scale, draw_info)

        return self.state.episode_rewards_list[0]

    def replay_window(
        self,
        # --- stop config
        timeout: int = -1,
        max_steps: int = -1,
        # --- progress
        enable_progress: bool = True,
        progress_start_time: int = 1,
        progress_interval_limit: int = 60 * 10,
        progress_env_info: bool = False,
        progress_worker_info: bool = True,
        progress_worker: int = 0,
        # --- other
        callbacks: List[CallbackType] = [],
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
        _is_test: bool = False,  # for test
    ):
        """TODO"""

        self.context.callbacks = callbacks[:]

        mode = PlayRenderModes.rgb_array

        # --- set context
        self.context.run_name = "main"
        self.context.distributed = False
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        # play config
        self.context.train_only = False
        self.context.shuffle_player = False
        # play info
        self.context.training = False
        self.context.render_mode = mode

        # init
        self.context.init(self)

        # --- progress ---
        if enable_progress:
            from srl.runner.callbacks.print_progress import PrintProgress

            self.context.callbacks.append(
                PrintProgress(
                    start_time=progress_start_time,
                    interval_limit=progress_interval_limit,
                    progress_env_info=progress_env_info,
                    progress_train_info=False,
                    progress_worker_info=progress_worker_info,
                    progress_worker=progress_worker,
                    progress_max_actor=5,
                    enable_eval=False,
                )
            )
        # ----------------

        from srl.runner.game_windows.replay_window import RePlayableGame

        window = RePlayableGame(self, parameter, remote_memory, _is_test)
        window.play()

    def play_terminal(
        self,
        players: List[Union[None, str, Tuple[str, dict], RLConfig]] = ["human"],
        # Rendering
        render_kwargs: dict = {},
        step_stop: bool = False,
        render_skip_step: bool = True,
        # --- stop config
        timeout: int = -1,
        max_steps: int = -1,
        # --- other
        callbacks: List[CallbackType] = [],
        parameter: Optional[RLParameter] = None,
        remote_memory: Optional[RLRemoteMemory] = None,
    ):
        """TODO"""

        self.context.callbacks = callbacks[:]
        self.config.players = players

        mode = PlayRenderModes.terminal

        # --- set context
        self.context.run_name = "main"
        self.context.distributed = False
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        # play config
        self.context.train_only = False
        self.context.shuffle_player = False
        # play info
        self.context.training = False
        self.context.render_mode = mode

        # init
        self.context.init(self)

        # --- rendering ---
        from srl.runner.callbacks.rendering import Rendering

        rendering = Rendering(
            mode=mode,
            kwargs=render_kwargs,
            step_stop=step_stop,
            render_skip_step=render_skip_step,
        )
        self.context.callbacks.append(rendering)
        # -----------------

        from .core import play

        if parameter is None:
            parameter = self.make_parameter()
        if remote_memory is None:
            remote_memory = self.make_remote_memory()
        play(self, parameter, remote_memory)

        return self.state.episode_rewards_list[0]

    def play_window(
        self,
        key_bind: Any = None,
        enable_remote_memory: bool = False,
        # --- stop config
        timeout: int = -1,
        max_steps: int = -1,
        # other
        callbacks: List[CallbackType] = [],
        _is_test: bool = False,  # for test
    ):
        """TODO"""

        self.context.callbacks = callbacks[:]

        mode = PlayRenderModes.rgb_array

        # --- set context
        self.context.run_name = "main"
        self.context.distributed = False
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        # play config
        self.context.train_only = False
        self.context.shuffle_player = False
        # play info
        self.context.training = enable_remote_memory
        self.context.render_mode = mode

        # init
        self.context.init(self)

        from srl.utils.common import is_packages_installed

        error_text = "This run requires installation of 'PIL', 'pygame'. "
        error_text += "(pip install pillow pygame)"
        assert is_packages_installed(["PIL", "pygame"]), error_text

        from srl.runner.game_windows.playable_game import PlayableGame

        game = PlayableGame(
            self,
            key_bind,
            enable_remote_memory=enable_remote_memory,
            _is_test=_is_test,
        )
        game.play()
