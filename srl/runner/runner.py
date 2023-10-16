import copy
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np

from srl.base.define import EnvObservationType, RLObservationType
from srl.base.env.config import EnvConfig
from srl.base.env.env_run import EnvRun
from srl.base.env.registration import make as make_env
from srl.base.rl.base import RLConfig, RLMemory, RLParameter, RLTrainer
from srl.base.rl.registration import make_memory, make_parameter, make_trainer, make_worker, make_worker_rulebase
from srl.base.rl.worker_run import WorkerRun
from srl.base.run import core as core_run
from srl.base.run.callback import CallbackData
from srl.base.run.data import RunContext, RunNameTypes, RunState
from srl.rl import dummy
from srl.runner.callback import CallbackType
from srl.utils import common
from srl.utils.serialize import convert_for_json

if TYPE_CHECKING:
    import psutil

logger = logging.getLogger(__name__)


@dataclass
class RunnerConfig:
    players: List[Union[None, str, Tuple[str, dict], RLConfig]] = field(default_factory=list)

    # --- dir
    # 基本となるディレクトリ、ファイル関係は、この配下に時刻でフォルダが作られ、その下が基準となる
    base_dir: str = "tmp"

    # --- stats
    enable_stats: bool = True

    # --- random
    seed: Optional[int] = None
    seed_enable_gpu: bool = True

    # --- device
    device_main: str = "AUTO"
    device_actors: Union[str, List[str]] = "AUTO"
    use_CUDA_VISIBLE_DEVICES: bool = True
    # tensorflow options
    tf_device_enable: bool = True
    tf_enable_memory_growth: bool = True

    def __post_init__(self):
        # --- stats
        self.used_psutil: bool = False
        self.used_nvidia: bool = False

    def to_dict(self) -> dict:
        dat: dict = convert_for_json(self.__dict__)
        return dat

    def copy(self) -> "RunnerConfig":
        return copy.deepcopy(self)


@dataclass
class RunnerMPData:
    env_config: EnvConfig
    rl_config: RLConfig
    config: RunnerConfig
    context: RunContext


@dataclass
class Runner(CallbackData):
    """実行環境を提供"""

    #: EnvConfigを指定（文字列のみのIDでも可能）
    name_or_env_config: Union[str, EnvConfig]
    #: RLConfigを指定, Noneの場合、dummyアルゴリズムが使われます
    rl_config: Optional[RLConfig] = None  # type: ignore , type

    config: Optional[RunnerConfig] = None  # type: ignore , type
    context: Optional[RunContext] = None  # type: ignore , type

    parameter: Optional[RLParameter] = None
    memory: Optional[RLMemory] = None

    # --- private(static class instance)
    # multiprocessing("spawn")ではプロセス毎に初期化される想定
    # pynvmlはプロセス毎に管理
    __is_init_process = False
    __framework = ""
    __used_device_tf = "/CPU"
    __used_device_torch = "cpu"
    __used_nvidia = False

    def __post_init__(self):
        # --- config
        if isinstance(self.name_or_env_config, str):
            self.env_config: EnvConfig = EnvConfig(self.name_or_env_config)
        else:
            self.env_config: EnvConfig = self.name_or_env_config
        if self.rl_config is None:
            self.rl_config: RLConfig = dummy.Config()
        if self.config is None:
            self.config: RunnerConfig = RunnerConfig()
        if self.context is None:
            self.context: RunContext = RunContext()

        self._env: Optional[EnvRun] = None
        self._trainer: Optional[RLTrainer] = None
        self._workers: Optional[List[WorkerRun]] = None

        self._history = []
        self._history_viewer = None
        self._history_on_memory_callback = None
        self._history_on_file_callback = None
        self._checkpoint_callback = None
        self._psutil_process: Optional["psutil.Process"] = None

    @property
    def env(self) -> EnvRun:
        return self.make_env()

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

    def set_save_dir(self, save_dir: str, setup_context: bool = False):
        """ディレクトリへの保存が必要な場合、そのディレクトリを指定します。
        ディレクトリへの保存は、historyやcheckpointがあります。

        Args:
            save_dir (str): 保存ディレクトリのパス。 Defaults to "tmp".
            setup_context (bool): contextも合わせてsetupします。 Defaults to False.
        """
        self.config.base_dir = save_dir
        if setup_context:
            self.context.setup(self.env_config, self.rl_config, self.config.base_dir)

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

    def set_stats(self, enable_stats: bool = True):
        """ハードウェアの統計情報に関する設定を指定します。

        Args:
            enable_stats (bool, optional): 統計情報の取得をするかどうか. Defaults to True.
        """
        self.config.enable_stats = enable_stats

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

    def save_memory(self, path: str, compress: bool = True, **kwargs):
        """save memory

        Args:
            path (str): save path
            compress (bool, optional): 圧縮するかどうか。圧縮はlzma形式です. Defaults to True.
        """
        self.make_memory().save(path, compress, **kwargs)

    def load_memory(self, path: str, **kwargs):
        """load memory"""
        self.make_memory().load(path, **kwargs)

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
        if self.parameter is None:
            if not self.rl_config.is_reset:
                self.rl_config.reset(self.make_env())
            self.parameter = make_parameter(self.rl_config, is_load=is_load)
            logger.info(f"make parameter: {self.parameter}")
        return self.parameter

    def make_memory(self, is_load: bool = True) -> RLMemory:
        self._init_process()
        if self.memory is None:
            if not self.rl_config.is_reset:
                self.rl_config.reset(self.make_env())
            self.memory = make_memory(self.rl_config, is_load=is_load)
            logger.info(f"make memory: {self.memory}")
        return self.memory

    def make_trainer(
        self,
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
        use_cache: bool = False,
    ) -> RLTrainer:
        if use_cache and self._trainer is not None:
            return self._trainer
        if parameter is None:
            parameter = self.make_parameter()
        if memory is None:
            memory = self.make_memory()
        if not self.rl_config.is_reset:
            self.rl_config.reset(self.make_env())
        trainer = make_trainer(self.rl_config, parameter, memory)
        if use_cache:
            self._trainer = trainer
        return trainer

    def make_worker(
        self,
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
    ) -> WorkerRun:
        if parameter is None:
            parameter = self.make_parameter()
        if memory is None:
            memory = self.make_memory()
        if not self.rl_config.is_reset:
            self.rl_config.reset(self.make_env())
        return make_worker(
            self.rl_config,
            self.make_env(),
            parameter,
            memory,
            self.context.distributed,
            self.context.actor_id,
        )

    def make_player(
        self,
        player: Union[None, str, RLConfig],
        worker_kwargs={},
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
    ) -> WorkerRun:
        env = self.make_env()

        # none はベース
        if player is None:
            return self.make_worker(parameter, memory)

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
            memory = make_memory(self.rl_config)
            worker = make_worker(
                player,
                env,
                parameter,
                memory,
                self.context.distributed,
                self.context.actor_id,
            )
            return worker

        raise ValueError(f"unknown worker: {player}")

    def make_players(
        self,
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
        use_cache: bool = False,
    ) -> List[WorkerRun]:
        if use_cache and self._workers is not None:
            return self._workers

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
            workers.append(self.make_player(p, kwargs, parameter, memory))

        if use_cache:
            self._workers = workers
        return workers

    # ------------------------------
    # process
    # ------------------------------
    def _init_process(self):
        self.__init_psutil()

        if Runner.__is_init_process:
            # 一度決定した値を使う
            # 別のframeworkを併用する場合の動作は未定義
            self.context.framework = Runner.__framework
            self.context.enable_tf_device = self.config.tf_device_enable
            self.context.used_device_tf = Runner.__used_device_tf
            self.context.used_device_torch = Runner.__used_device_torch
            self.rl_config._used_device_tf = self.context.used_device_tf
            self.rl_config._used_device_torch = self.context.used_device_torch
            self.config.used_nvidia = Runner.__used_nvidia
            return

        self.__init_nvidia()
        self.__init_device()

        Runner.__framework = self.context.framework
        Runner.__used_device_tf = self.context.used_device_tf
        Runner.__used_device_torch = self.context.used_device_torch
        Runner.__used_nvidia = self.config.used_nvidia
        Runner.__is_init_process = True

    # --- system profile
    def __init_nvidia(self):
        if not self.config.enable_stats:
            return

        self.config.used_nvidia = False
        if common.is_package_installed("pynvml"):
            try:
                import pynvml

                pynvml.nvmlInit()
                self.config.used_nvidia = True

            except Exception as e:
                import traceback

                logger.debug(traceback.format_exc())
                logger.info(e)

    def close_nvidia(self):
        if Runner.__used_nvidia:
            Runner.__used_nvidia = False
            self.config.used_nvidia = False
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:
                import traceback

                logger.info(traceback.format_exc())

    def read_nvml(self) -> List[Tuple[int, float, float]]:
        if not self.config.used_nvidia:
            return []
        import pynvml

        gpu_num = pynvml.nvmlDeviceGetCount()
        gpus = []
        for device_id in range(gpu_num):
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpus.append((device_id, float(rate.gpu), float(rate.memory)))
        return gpus

    # --- device
    def set_device(
        self,
        device_trainer: str = "AUTO",
        device_actors: Union[str, List[str]] = "AUTO",
        use_CUDA_VISIBLE_DEVICES: bool = True,
        tf_device_enable: bool = True,
        tf_enable_memory_growth: bool = True,
    ):
        """set device.

        "AUTO",""    : Automatic assignment.
        "CPU","CPU:0": Use CPU.
        "GPU","GPU:0": Use GPU.

        AUTO assign
          sequence
            trainer: GPU -> CPU
            actors : not use
          distribute
            trainer: GPU -> CPU
            actors : CPU

        Args:
            device_main (str, optional): mainのdeviceを指定します。分散学習を用いない場合、これだけが使用されます. Defaults to "AUTO".
            device_mp_trainer (str, optional): 分散学習時のtrainerが使うdeviceを指定します. Defaults to "AUTO".
            device_mp_actors (Union[str, List[str]], optional): 分散学習時のactorが使うdeviceを指定します. Defaults to "AUTO".
            use_CUDA_VISIBLE_DEVICES (bool, optional): CPUの場合 CUDA_VISIBLE_DEVICES を-1にする. Defaults to True.
            tf_device_enable (bool, optional): tensorflowにて、 'with tf.device()' を使用する. Defaults to True.
            tf_enable_memory_growth (bool, optional): tensorflowにて、'set_memory_growth(True)' を実行する. Defaults to True.
        """
        if Runner.__is_init_process:
            logger.warning("Device cannot be changed after initialization.")
            return

        self.config.device_main = device_trainer
        self.config.device_actors = device_actors
        self.config.use_CUDA_VISIBLE_DEVICES = use_CUDA_VISIBLE_DEVICES
        self.config.tf_device_enable = tf_device_enable
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
        actor_id = self.context.actor_id

        # logger
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            cuda_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            logger.info(f"[{run_name}] CUDA_VISIBLE_DEVICES='{cuda_devices}'")
        else:
            logger.info(f"[{run_name}] CUDA_VISIBLE_DEVICES is not define.")

        # --- check device ---
        if run_name == RunNameTypes.main:
            device = self.config.device_main.upper()
            if device == "":
                device = "AUTO"
        elif run_name == RunNameTypes.actor:
            if isinstance(self.config.device_actors, str):
                device = self.config.device_actors.upper()
            else:
                device = self.config.device_actors[actor_id].upper()
            if device in ["", "AUTO"]:
                device = "CPU"
        else:
            device = "CPU"
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
            if "CPU:" in device:
                t = device.split(":")
                self.context.used_device_tf = f"/CPU:{t[1]}"
                self.context.used_device_torch = f"cpu:{t[1]}"
            else:
                self.context.used_device_tf = "/CPU"
                self.context.used_device_torch = "cpu"
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

        self.context.enable_tf_device = self.config.tf_device_enable
        self.rl_config._used_device_tf = self.context.used_device_tf
        self.rl_config._used_device_torch = self.context.used_device_torch
        logger.info(f"[{run_name}] Initialized device.")

    # ------------------------------
    # psutil
    # ------------------------------
    def __init_psutil(self):
        if not self.config.enable_stats:
            return
        if self._psutil_process is not None:
            return

        self.config.used_psutil = False
        if common.is_package_installed("psutil"):
            try:
                import psutil

                self._psutil_process = psutil.Process()
                self.config.used_psutil = True
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

    # ------------------------------
    # utility
    # ------------------------------
    def get_env_init_state(self, encode: bool = True) -> Union[EnvObservationType, RLObservationType]:
        env = self.make_env()
        env.reset()
        state = env.state
        if encode:
            worker = self.make_worker()
            state = worker.state_encode(state, env, append_recent_state=False)
        return state

    def copy(self, copy_context: bool = False, env_share: bool = False) -> "Runner":
        runner = Runner(self.env_config, self.rl_config)
        runner.config = self.config.copy()
        if copy_context:
            runner.context = self.context.copy()
        if env_share:
            runner._env = self._env
        return runner

    def create_mp_data(self) -> RunnerMPData:
        return RunnerMPData(self.env_config, self.rl_config, self.config, self.context)

    def print_config(self):
        import pprint

        print(f"env\n{pprint.pformat(self.env_config.to_dict())}")
        print(f"rl\n{pprint.pformat(self.rl_config.to_dict())}")
        print(f"context\n{pprint.pformat(self.context.to_dict())}")

    # ------------------------------
    # eval
    # ------------------------------
    def create_eval_runner(
        self,
        env_share: bool,
        eval_episode: int,
        eval_timeout: int,
        eval_max_steps: int,
        eval_players: List[Union[None, str, Tuple[str, dict], RLConfig]],
        eval_shuffle_player: bool,
        enable_tf_device: bool,
        used_device_tf: str,
        used_device_torch: str,
        eval_callbacks: List[CallbackType],
    ) -> "Runner":
        eval_runner = self.copy(copy_context=False, env_share=env_share)
        # config
        eval_runner.config.players = eval_players
        # context
        c = RunContext()
        c.callbacks = eval_callbacks[:]  # type: ignore , type ok
        c.run_name = RunNameTypes.eval
        # stop
        c.max_episodes = eval_episode
        c.timeout = eval_timeout
        c.max_steps = eval_max_steps
        # play
        c.shuffle_player = eval_shuffle_player
        c.training = False
        c.seed = None
        # device
        c.enable_tf_device = enable_tf_device
        c.used_device_tf = used_device_tf
        c.used_device_torch = used_device_torch
        c.setup(self.env_config, self.rl_config)
        eval_runner.context = c
        return eval_runner

    def callback_play_eval(self, parameter: RLParameter):
        state = core_run.play(
            self.context,
            self.make_env(),
            workers=self.make_players(parameter=parameter, use_cache=True),
            trainer=self.make_trainer(parameter=parameter, use_cache=True),
        )
        return state.episode_rewards_list

    # ---------------------------------------------
    # run
    # ---------------------------------------------
    def core_play(
        self,
        trainer_only: bool = False,
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
        trainer: Optional[RLTrainer] = None,
    ) -> RunState:
        # --- random ---
        if self.config.seed is not None:
            common.set_seed(self.config.seed, self.config.seed_enable_gpu)
            self.context.seed = self.config.seed
        # --------------

        if trainer is None:
            trainer = self.make_trainer(parameter=parameter, memory=memory)

        # --- play ----
        if not trainer_only:
            state = core_run.play(
                self.context,
                self.make_env(),
                self.make_players(parameter=parameter, memory=memory),
                trainer,
                self,
            )
        else:
            state = core_run.play_trainer_only(self.context, trainer, self)
        # ----------------
        return state
