import copy
import datetime
import glob
import logging
import os
import pickle
import re
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple, Union, cast

import numpy as np

from srl.base.define import EnvObservationType, RLObservationType
from srl.base.env.config import EnvConfig
from srl.base.env.env_run import EnvRun
from srl.base.env.registration import make as make_env
from srl.base.rl.base import RLConfig, RLMemory, RLParameter, RLTrainer
from srl.base.rl.registration import make_memory, make_parameter, make_trainer, make_worker
from srl.base.rl.worker_run import WorkerRun
from srl.base.run import core as base_run
from srl.base.run.callback import RunCallback, TrainerCallback
from srl.base.run.context import RLWorkerType, RunContext, RunNameTypes, StrWorkerType
from srl.rl import dummy
from srl.runner.callback import GameCallback, RunnerCallback
from srl.utils import common
from srl.utils.serialize import convert_for_json

if TYPE_CHECKING:
    import psutil

    from srl.runner.callbacks.history_viewer import HistoryViewer, HistoryViewers

logger = logging.getLogger(__name__)


CallbackType = Union[RunCallback, TrainerCallback, RunnerCallback, GameCallback]


@dataclass
class RunnerConfig:
    # --- mp
    dist_queue_capacity: int = 1000
    trainer_parameter_send_interval: int = 1  # sec
    actor_parameter_sync_interval: int = 1  # sec
    dist_enable_prepare_sample_batch: bool = False
    dist_enable_trainer_thread: bool = True
    dist_enable_actor_thread: bool = True
    device_actors: Union[str, List[str]] = "CPU"

    # --- stats
    enable_stats: bool = False

    # --- random
    seed: Optional[int] = None
    seed_enable_gpu: bool = True

    # --- device
    device: str = "AUTO"
    set_CUDA_VISIBLE_DEVICES_if_CPU: bool = True
    # tensorflow options
    tf_device_enable: bool = True
    tf_enable_memory_growth: bool = True

    def __post_init__(self):
        self.enable_wkdir: bool = False
        self.wkdir: str = ""
        self.wkdir1: str = ""
        self.wkdir2: str = ""

    def to_dict(self) -> dict:
        dat: dict = convert_for_json(self.__dict__)
        return dat

    def copy(self) -> "RunnerConfig":
        return copy.deepcopy(self)


@dataclass
class TaskConfig:
    config: RunnerConfig
    context: RunContext
    callbacks: List[CallbackType]


@dataclass
class Runner:
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
    __setup_device = False
    __framework = ""
    __used_device_tf = "/CPU"
    __used_device_torch = "cpu"
    # pynvmlはプロセス毎に管理
    __used_nvidia = None

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
            self.context: RunContext = RunContext(self.env_config, self.rl_config)
        self.context_controller = self.context.create_controller()

        self._env: Optional[EnvRun] = None
        self._trainer: Optional[RLTrainer] = None
        self._workers: Optional[List[WorkerRun]] = None

        self._history_on_file_kwargs: Optional[dict] = None
        self._history_on_memory_kwargs: Optional[dict] = None
        self._checkpoint_kwargs: Optional[dict] = None
        self.history_viewer: Optional["HistoryViewer"] = None

        self._progress_kwargs: dict = {}

        self._is_setup_psutil: bool = False
        self._psutil_process: Optional["psutil.Process"] = None

    @property
    def env(self) -> EnvRun:
        return self.make_env()

    # ------------------------------
    # set config
    # ------------------------------
    def set_players(self, players: List[Union[None, StrWorkerType, RLWorkerType]] = []):
        """multi player option
        マルチプレイヤーゲームでのプレイヤーを指定します。

        None                : use rl_config worker
        str                 : Registered RuleWorker
        Tuple[str, dict]    : Registered RuleWorker(Pass kwargs argument)
        RLConfig            : use RLConfig worker
        Tuple[RLConfig, Any]: use RLConfig worker(Parameter)

        Args:
            players: マルチプレイヤーゲームにおけるプレイヤーを表した配列

        """
        self.context.players = players

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

    def enable_stats(self):
        """ハードウェアの統計情報に関する設定を有効にします。"""
        self.config.enable_stats = True

    def disable_stats(self):
        """ハードウェアの統計情報に関する設定を無効にします。"""
        self.config.enable_stats = False

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
        self._setup_process()
        if self.parameter is None:
            if not self.rl_config._is_setup:
                self.rl_config.setup(self.make_env())
            self.parameter = make_parameter(self.rl_config, is_load=is_load)
            logger.info(f"make parameter: {self.parameter}")
        return self.parameter

    def make_memory(self, is_load: bool = True) -> RLMemory:
        self._setup_process()
        if self.memory is None:
            if not self.rl_config._is_setup:
                self.rl_config.setup(self.make_env())
            self.memory = make_memory(self.rl_config, is_load=is_load)
            logger.info(f"make memory: {self.memory}")
        return self.memory

    def make_trainer(
        self,
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
        use_cache: bool = False,
    ) -> RLTrainer:
        self._setup_process()
        if use_cache and self._trainer is not None:
            return self._trainer
        if parameter is None:
            parameter = self.make_parameter()
        if memory is None:
            memory = self.make_memory()
        if not self.rl_config._is_setup:
            self.rl_config.setup(self.make_env())
        trainer = make_trainer(self.rl_config, parameter, memory)
        if use_cache:
            self._trainer = trainer
        return trainer

    def make_worker(
        self,
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
    ) -> WorkerRun:
        self._setup_process()
        env = self.make_env()
        self.rl_config.setup(env)
        if parameter is None:
            parameter = self.make_parameter()
        if memory is None:
            memory = self.make_memory()
        return make_worker(
            self.rl_config,
            env,
            parameter,
            memory,
            self.context.distributed,
            self.context.actor_id,
        )

    def make_workers(
        self,
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
        use_cache: bool = False,
    ) -> List[WorkerRun]:
        if use_cache and self._workers is not None:
            return self._workers

        if parameter is None:
            parameter = self.make_parameter()
        if memory is None:
            memory = self.make_memory()
        workers = self.context_controller.make_workers(self.make_env(), parameter, memory)

        if use_cache:
            self._workers = workers
        return workers

    # ------------------------------
    # process
    # ------------------------------
    def _setup_process(self):
        if self.config.enable_stats:
            Runner.setup_nvidia()

        # --- device
        if not Runner.__setup_device:
            framework = self.rl_config.get_use_framework()
            if framework == "":
                return
            device = self.get_device(self.context.run_name, self.context.actor_id)
            used_device_tf, used_device_torch = Runner.setup_device(
                framework,
                device,
                self.config.set_CUDA_VISIBLE_DEVICES_if_CPU,
                self.config.tf_enable_memory_growth,
            )
            self.context_controller.set_device(used_device_tf, used_device_torch)

    # ------------------------------
    # nvidia
    # ------------------------------
    @staticmethod
    def setup_nvidia():
        if Runner.__used_nvidia is not None:
            return
        Runner.__used_nvidia = False
        if common.is_package_installed("pynvml"):
            try:
                import pynvml

                pynvml.nvmlInit()
                Runner.__used_nvidia = True

            except Exception as e:
                logger.debug(traceback.format_exc())
                logger.info(e)

    @staticmethod
    def close_nvidia():
        if Runner.__used_nvidia is not None and Runner.__used_nvidia:
            Runner.__used_nvidia = None
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:
                logger.info(traceback.format_exc())

    @staticmethod
    def read_nvml() -> List[Tuple[int, float, float]]:
        if Runner.__used_nvidia is None:
            return []
        if not Runner.__used_nvidia:
            return []

        import pynvml

        gpu_num = pynvml.nvmlDeviceGetCount()
        gpus = []
        for device_id in range(gpu_num):
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpus.append((device_id, float(rate.gpu), float(rate.memory)))
        return gpus

    # ------------------------------
    # device
    # ------------------------------
    def set_device(
        self,
        device: str = "AUTO",
        set_CUDA_VISIBLE_DEVICES_if_CPU: bool = True,
        tf_device_enable: bool = True,
        tf_enable_memory_growth: bool = True,
    ):
        """set device.

        "AUTO",""    : Automatic assignment.
        "CPU","CPU:0": Use CPU.
        "GPU","GPU:0": Use GPU.

        Args:
            device (str, optional): mainのdeviceを指定します。分散学習を用いない場合、これだけが使用されます. Defaults to "AUTO".
            set_CUDA_VISIBLE_DEVICES_if_CPU (bool, optional): CPUの場合 CUDA_VISIBLE_DEVICES を-1にする. Defaults to True.
            tf_device_enable (bool, optional): tensorflowにて、 'with tf.device()' を使用する. Defaults to True.
            tf_enable_memory_growth (bool, optional): tensorflowにて、'set_memory_growth(True)' を実行する. Defaults to True.
        """
        if Runner.__setup_device:
            logger.warning("Device cannot be changed after initialization.")
            return

        self.config.device = device
        self.config.set_CUDA_VISIBLE_DEVICES_if_CPU = set_CUDA_VISIBLE_DEVICES_if_CPU
        self.config.tf_device_enable = tf_device_enable
        self.config.tf_enable_memory_growth = tf_enable_memory_growth

    @staticmethod
    def setup_device(
        framework: str,
        device: str,
        set_CUDA_VISIBLE_DEVICES_if_CPU: bool = True,
        tf_enable_memory_growth: bool = True,
    ) -> Tuple[str, str]:
        device = device.upper()

        # frameworkは "" の場合何もしない(フラグも立てない)
        if framework == "":
            return "/CPU", "cpu"
        if framework == "tf":
            framework = "tensorflow"
        assert framework in ["tensorflow", "torch"], "Framework can specify 'tensorflow' or 'torch'."

        if Runner.__setup_device:
            if Runner.__framework != framework:
                logger.warning(
                    f"Initialization with a different framework is not assumed. {Runner.__framework} != {framework}"
                )
            return Runner.__used_device_tf, Runner.__used_device_torch

        # logger
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            cuda_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            logger.info(f"[device] CUDA_VISIBLE_DEVICES='{cuda_devices}'")
        else:
            logger.info("[device] CUDA_VISIBLE_DEVICES is not define.")

        # --- CUDA_VISIBLE_DEVICES ---
        if set_CUDA_VISIBLE_DEVICES_if_CPU:
            if "CPU" in device:
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                logger.info("[device] set CUDA_VISIBLE_DEVICES=-1")
            else:
                # CUDA_VISIBLE_DEVICES が -1 の場合のみ削除する
                if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "-1":
                    del os.environ["CUDA_VISIBLE_DEVICES"]
                    logger.info("[device] del CUDA_VISIBLE_DEVICES")
        # -----------------------------

        # --- tf memory growth ---
        # Tensorflow,GPU がある場合に実施(CPUにしてもなぜかGPUの初期化は走る場合あり)
        if framework == "tensorflow" and tf_enable_memory_growth:
            try:
                import tensorflow as tf

                gpu_devices = tf.config.list_physical_devices("GPU")
                for d in gpu_devices:
                    logger.info(f"[device] (tf) set_memory_growth({d.name}, True)")
                    tf.config.experimental.set_memory_growth(d, True)
            except Exception:
                s = "[device] (tf) 'set_memory_growth' failed."
                s += " Also consider 'Runner.setup_device(tf_enable_memory_growth=False)'."
                print(s)
                raise
        # -----------------------

        if "CPU" in device:
            # --- CPU ---
            if "CPU:" in device:
                t = device.split(":")
                used_device_tf = f"/CPU:{t[1]}"
                used_device_torch = f"cpu:{t[1]}"
            else:
                used_device_tf = "/CPU"
                used_device_torch = "cpu"
        else:
            used_device_tf = "/CPU"
            used_device_torch = "cpu"

            # --- GPU (AUTOの場合もあり) ---
            if framework == "tensorflow":
                # --- tensorflow GPU check
                import tensorflow as tf

                gpu_devices = tf.config.list_physical_devices("GPU")
                if len(gpu_devices) == 0:
                    if "GPU" in device:
                        logger.warning(f"[device] (tf) GPU is not found. {tf.config.list_physical_devices()}")

                    used_device_tf = "/CPU"

                else:
                    logger.info(f"[device] (tf) gpu device: {len(gpu_devices)}")

                    if "GPU:" in device:
                        t = device.split(":")
                        used_device_tf = f"/GPU:{t[1]}"
                    else:
                        used_device_tf = "/GPU"

            if framework == "torch":
                # --- torch GPU check
                import torch

                if torch.cuda.is_available():
                    logger.info(f"[device] (torch) gpu device: {torch.cuda.get_device_name()}")

                    if "GPU:" in device:
                        t = device.split(":")
                        used_device_torch = f"cuda:{t[1]}"
                    else:
                        used_device_torch = "cuda"
                else:
                    if "GPU" in device:
                        logger.warning("[device] (torch) GPU is not found.")

                    used_device_torch = "cpu"
        # -------------------------

        Runner.__setup_device = True
        Runner.__framework = framework
        Runner.__used_device_tf = used_device_tf
        Runner.__used_device_torch = used_device_torch

        logger.info("[device] Initialized device.")
        return used_device_tf, used_device_torch

    def get_device(self, run_name: RunNameTypes, actor_id: int) -> str:
        if run_name == RunNameTypes.main or run_name == RunNameTypes.trainer:
            device = self.config.device.upper()
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
        return device

    # ------------------------------
    # psutil
    # ------------------------------
    def read_psutil(self) -> Tuple[float, float]:
        if not self._is_setup_psutil:
            self._is_setup_psutil = True
            self._psutil_process = None
            if common.is_package_installed("psutil"):
                try:
                    import psutil

                    self._psutil_process = psutil.Process()
                except Exception as e:
                    import traceback

                    logger.debug(traceback.format_exc())
                    logger.info(e)

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

    def copy(self, env_share: bool) -> "Runner":
        runner = Runner(
            self.env_config.copy(),
            self.rl_config.copy(),
            self.config.copy(),
            self.context_controller.copy(),
        )
        if env_share:
            runner._env = self._env
        return runner

    def create_task_config(self, callbacks: List[CallbackType], exclude_callbacks: List[str] = []) -> TaskConfig:
        c = []
        for c2 in callbacks:
            f = True
            for e in exclude_callbacks:
                if e in c2.__class__.__name__:
                    f = False
                    break
            if f:
                c.append(c2)

        return TaskConfig(
            self.config.copy(),
            self.context_controller.copy(),
            pickle.loads(pickle.dumps(c)),
        )

    def print_config(self):
        import pprint

        print(f"env\n{pprint.pformat(self.env_config.to_dict())}")
        print(f"rl\n{pprint.pformat(self.rl_config.to_dict())}")
        print(f"context\n{pprint.pformat(self.context_controller.to_dict())}")

    # ------------------------------
    # eval
    # ------------------------------
    def create_eval_runner(
        self,
        env_share: bool,
        eval_episode: int,
        eval_timeout: float,
        eval_max_steps: int,
        eval_players: List[Union[None, StrWorkerType, RLWorkerType]],
        eval_shuffle_player: bool,
    ) -> "Runner":
        r = self.copy(env_share)

        # context
        r.context.players = eval_players
        r.context.run_name = RunNameTypes.eval
        # stop
        r.context.max_episodes = eval_episode
        r.context.timeout = eval_timeout
        r.context.max_steps = eval_max_steps
        # play
        r.context.shuffle_player = eval_shuffle_player
        r.context.training = False
        r.context.seed = None  # mainと競合するのでNone
        return r

    def callback_play_eval(self, parameter: RLParameter):
        env = self.make_env()
        memory = self.make_memory(is_load=False)
        state = base_run.play(
            self.context,
            env,
            parameter=parameter,
            memory=memory,
            workers=self.make_workers(parameter, memory, use_cache=True),
            trainer=None,
        )
        return state.episode_rewards_list

    # ------------------------------
    # wkdir/
    #  └ [env]_[rl]/
    #    ├ checkpoint/
    #    └ run_YYYYMMDD_HHMMSS/
    #      ├ history/
    #      └ ConfigFiles
    # ------------------------------
    def setup_wkdir(
        self,
        wkdir: str = "tmp",
        checkpoint_interval: int = 60 * 20,
        checkpoint_eval: bool = True,
        checkpoint_eval_env_sharing: bool = False,
        checkpoint_eval_episode: int = 1,
        checkpoint_eval_timeout: float = -1,
        checkpoint_eval_max_steps: int = -1,
        checkpoint_eval_players: List[Union[None, StrWorkerType, RLWorkerType]] = [],
        checkpoint_eval_shuffle_player: bool = False,
        history_interval: int = 10,
        history_write_system: bool = False,
        history_eval: bool = False,
        history_eval_env_sharing: bool = False,
        history_eval_episode: int = 1,
        history_eval_timeout: float = -1,
        history_eval_max_steps: int = -1,
        history_eval_players: List[Union[None, StrWorkerType, RLWorkerType]] = [],
        history_eval_shuffle_player: bool = False,
    ):
        """
        作業ディレクトリを作成し、学習状況を保存します。
        """
        self.config.wkdir = wkdir
        dir_name = f"{self.context.env_config.name}_{self.context.rl_config.getName()}"
        dir_name = re.sub(r'[\\/:?."<>\|]', "_", dir_name)
        self.config.wkdir1 = os.path.join(wkdir, dir_name)
        os.makedirs(self.config.wkdir1, exist_ok=True)
        logger.info(f"create wkdir: {self.config.wkdir1}")

        self.set_checkpoint(
            os.path.join(self.config.wkdir1, "checkpoint"),
            checkpoint_interval,
            checkpoint_eval,
            checkpoint_eval_env_sharing,
            checkpoint_eval_episode,
            checkpoint_eval_timeout,
            checkpoint_eval_max_steps,
            checkpoint_eval_players,
            checkpoint_eval_shuffle_player,
        )
        self.set_history_on_file(
            self.config.wkdir1,
            history_interval,
            True,
            history_write_system,
            history_eval,
            history_eval_env_sharing,
            history_eval_episode,
            history_eval_timeout,
            history_eval_max_steps,
            history_eval_players,
            history_eval_shuffle_player,
        )
        self.config.enable_wkdir = True

    # ------------------------------
    # progress
    # ------------------------------
    def set_progress_options(
        self,
        start_time: int = 1,
        interval_limit: int = 60 * 10,
        single_line=True,
        env_info: bool = False,
        train_info: bool = True,
        worker_info: bool = True,
        worker: int = 0,
        max_actor: int = 5,
        # --- eval
        eval_env_sharing: bool = False,
        eval_shuffle_player: bool = False,
    ):
        """progress options

        Args:
            start_time (int, optional): 最初に進捗を表示する秒数. Defaults to 1.
            interval_limit (int, optional): 進捗を表示する最大の間隔（秒）. Defaults to 60*10.
            single_line (bool, optional): 表示を1lineにするか. Defaults to False.
            env_info (bool, optional): 進捗表示にenv infoを表示するか. Defaults to False.
            train_info (bool, optional): 進捗表示にtrain infoを表示するか. Defaults to True.
            worker_info (bool, optional): 進捗表示にworker infoを表示するか. Defaults to True.
            worker (int, optional): 進捗表示に表示するworker index. Defaults to 0.
            max_actor (int, optional): 進捗表示に表示するworker数. Defaults to 5.
            eval_env_sharing (bool, optional): 評価時に学習時のenvを共有します. Defaults to False.
            eval_shuffle_player (bool, optional): 評価時にplayersをシャッフルするか. Defaults to False.
        """
        self._progress_kwargs = dict(
            start_time=start_time,
            interval_limit=interval_limit,
            single_line=single_line,
            progress_env_info=env_info,
            progress_train_info=train_info,
            progress_worker_info=worker_info,
            progress_worker=worker,
            progress_max_actor=max_actor,
            eval_env_sharing=eval_env_sharing,
            eval_shuffle_player=eval_shuffle_player,
        )

    # ------------------------------
    # history
    # ------------------------------
    @staticmethod
    def load_history(history_dir: str) -> "HistoryViewer":
        from srl.runner.callbacks.history_viewer import HistoryViewer

        return HistoryViewer(history_dir)

    @staticmethod
    def load_histories(history_dirs: List[str]) -> "HistoryViewers":
        from srl.runner.callbacks.history_viewer import HistoryViewers

        return HistoryViewers(history_dirs)

    def get_history(self) -> "HistoryViewer":
        from srl.runner.callbacks.history_viewer import HistoryViewer

        if self.config.enable_wkdir:
            # 最後の結果を探す
            dirs = []
            for d in glob.glob(os.path.join(self.config.wkdir1, "run_*")):
                try:
                    _s = os.path.basename(d).split("_")
                    date = datetime.datetime.strptime(f"{_s[1]}_{_s[2]}", "%Y%m%d_%H%M%S")
                    dirs.append([date, d])
                except Exception:
                    logger.warning(traceback.format_exc())

            if len(dirs) == 0:
                return HistoryViewer()
            dirs.sort()

            _history_viewer = HistoryViewer()
            _history_viewer.load(dirs[-1][1])
            return _history_viewer

        assert self.history_viewer is not None
        return self.history_viewer

    def set_history_on_memory(
        self,
        interval: int = 1,
        enable_eval: bool = False,
        eval_env_sharing: bool = False,
        eval_episode: int = 1,
        eval_timeout: float = -1,
        eval_max_steps: int = -1,
        eval_players: List[Union[None, StrWorkerType, RLWorkerType]] = [],
        eval_shuffle_player: bool = False,
    ):
        """学習履歴を保存する設定を指定します。

        Args:
            interval (int, optional): 学習履歴を保存する間隔(秒). Defaults to 1.
            enable_eval (bool, optional): 学習履歴の保存時に評価用のシミュレーションを実行します. Defaults to False.
            eval_env_sharing (bool, optional): 評価時に学習時のenvを共有します. Defaults to False.
            eval_episode (int, optional): 評価時のエピソード数. Defaults to 1.
            eval_timeout (int, optional): 評価時の1エピソードの制限時間. Defaults to -1.
            eval_max_steps (int, optional): 評価時の1エピソードの最大ステップ数. Defaults to -1.
            eval_players (List[Union[None, str, Tuple[str, dict], RLConfig]], optional): 評価時のplayers. Defaults to [].
            eval_shuffle_player (bool, optional): 評価時にplayersをシャッフルするか. Defaults to False.
        """

        self._history_on_memory_kwargs = dict(
            interval=interval,
            enable_eval=enable_eval,
            eval_env_sharing=eval_env_sharing,
            eval_episode=eval_episode,
            eval_timeout=eval_timeout,
            eval_max_steps=eval_max_steps,
            eval_players=eval_players,
            eval_shuffle_player=eval_shuffle_player,
        )

    def set_history_on_file(
        self,
        save_dir: str,
        interval: int = 1,
        add_history: bool = False,
        write_system: bool = False,
        enable_eval: bool = False,
        eval_env_sharing: bool = False,
        eval_episode: int = 1,
        eval_timeout: float = -1,
        eval_max_steps: int = -1,
        eval_players: List[Union[None, StrWorkerType, RLWorkerType]] = [],
        eval_shuffle_player: bool = False,
    ):
        """学習履歴を保存する設定を指定します。

        Args:
            save_dir (str): 保存するディレクトリ
            interval (int, optional): 学習履歴を保存する間隔(秒). Defaults to 1.
            add_history (bool, optional): 追記で学習履歴を保存. Defaults to False.
            write_system (bool, optional): CPU/memory情報も保存. Defaults to False.
            enable_eval (bool, optional): 学習履歴の保存時に評価用のシミュレーションを実行します. Defaults to False.
            eval_env_sharing (bool, optional): 評価時に学習時のenvを共有します. Defaults to False.
            eval_episode (int, optional): 評価時のエピソード数. Defaults to 1.
            eval_timeout (int, optional): 評価時の1エピソードの制限時間. Defaults to -1.
            eval_max_steps (int, optional): 評価時の1エピソードの最大ステップ数. Defaults to -1.
            eval_players (List[Union[None, str, Tuple[str, dict], RLConfig]], optional): 評価時のplayers. Defaults to [].
            eval_shuffle_player (bool, optional): 評価時にplayersをシャッフルするか. Defaults to False.
        """

        self._history_on_file_kwargs = dict(
            save_dir=save_dir,
            interval=interval,
            add_history=add_history,
            write_system=write_system,
            enable_eval=enable_eval,
            eval_env_sharing=eval_env_sharing,
            eval_episode=eval_episode,
            eval_timeout=eval_timeout,
            eval_max_steps=eval_max_steps,
            eval_players=eval_players,
            eval_shuffle_player=eval_shuffle_player,
        )
        if write_system:
            self.enable_stats()

    def disable_history(self):
        self._history_on_memory_kwargs = None
        self._history_on_file_kwargs = None

    # ------------------------------
    # checkpoint
    # ------------------------------
    def load_checkpoint(self, save_dir: str):
        from srl.runner.callbacks.checkpoint import Checkpoint

        path = Checkpoint.get_parameter_path(save_dir)
        if os.path.isfile(path):
            try:
                self.make_parameter(is_load=False).load(path)
                logger.info(f"Checkpoint parameter loaded: {path}")
            except Exception as e:
                logger.info(e)
                logger.warning(f"Failed to load parameter. Run without loading. {path}")
        else:
            logger.info(f"Checkpoint parameter is not found: {path}")

    def set_checkpoint(
        self,
        save_dir: str,
        interval: int = 60 * 20,
        enable_eval: bool = True,
        eval_env_sharing: bool = False,
        eval_episode: int = 1,
        eval_timeout: float = -1,
        eval_max_steps: int = -1,
        eval_players: List[Union[None, StrWorkerType, RLWorkerType]] = [],
        eval_shuffle_player: bool = False,
    ):
        """一定間隔でモデルを保存します。

        Args:
            save_dir (int): 保存するディレクトリ
            interval (int, optional): 保存する間隔（秒）. Defaults to 60*20sec.
            enable_eval (bool, optional): モデル保存時に評価用のシミュレーションを実行します. Defaults to False.
            eval_env_sharing (bool, optional): 評価時に学習時のenvを共有します. Defaults to False.
            eval_episode (int, optional): 評価時のエピソード数. Defaults to 1.
            eval_timeout (int, optional): 評価時の1エピソードの制限時間. Defaults to -1.
            eval_max_steps (int, optional): 評価時の1エピソードの最大ステップ数. Defaults to -1.
            eval_players (List[Union[None, str, Tuple[str, dict], RLConfig]], optional): 評価時のplayers. Defaults to [].
            eval_shuffle_player (bool, optional): 評価時にplayersをシャッフルするか. Defaults to False.
        """

        self._checkpoint_kwargs = dict(
            save_dir=save_dir,
            interval=interval,
            enable_eval=enable_eval,
            eval_env_sharing=eval_env_sharing,
            eval_episode=eval_episode,
            eval_timeout=eval_timeout,
            eval_max_steps=eval_max_steps,
            eval_players=eval_players,
            eval_shuffle_player=eval_shuffle_player,
        )

    def disable_checkpoint(self):
        self._checkpoint_kwargs = None

    # ---------------------------------------------
    # run
    # ---------------------------------------------
    def _base_run_play_before(
        self,
        enable_checkpoint_load: bool,
        enable_checkpoint: bool,
        enable_history_on_memory: bool,
        enable_history_on_file: bool,
        callbacks: List[CallbackType],
    ):
        self._callbacks = callbacks[:]

        # --- wkdir ---
        if self.config.enable_wkdir:
            dir_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.config.wkdir2 = os.path.join(self.config.wkdir1, f"run_{dir_name}")
        # -------------

        # --- checkpoint load ---
        if enable_checkpoint_load and self.config.enable_wkdir:
            checkpoint_dir = os.path.join(self.config.wkdir1, "checkpoint")
            self.load_checkpoint(checkpoint_dir)
        # -----------------------

        # --- checkpoint ---
        if enable_checkpoint and self._checkpoint_kwargs is not None:
            from srl.runner.callbacks.checkpoint import Checkpoint

            callbacks.append(Checkpoint(**self._checkpoint_kwargs))
            logger.info(f"add callback Checkpoint: {self._checkpoint_kwargs['save_dir']}")
        # -------------------

        # --- history ---
        if enable_history_on_memory and self._history_on_memory_kwargs is not None:
            from srl.runner.callbacks.history_on_memory import HistoryOnMemory

            callbacks.append(HistoryOnMemory(**self._history_on_memory_kwargs))
            logger.info("add callback HistoryOnMemory")

        if enable_history_on_file and self._history_on_file_kwargs is not None:
            from srl.runner.callbacks.history_on_file import HistoryOnFile

            callbacks.append(HistoryOnFile(**self._history_on_file_kwargs))
            logger.info(f"add callback HistoryOnFile: {self._history_on_file_kwargs['save_dir']}")
        # ---------------

        # --- callback
        [cast(RunnerCallback, c).on_runner_start(self) for c in callbacks if issubclass(c.__class__, RunnerCallback)]

    def _base_run_play_after(self, callbacks: List[CallbackType]):
        # --- callback
        [cast(RunnerCallback, c).on_runner_end(self) for c in callbacks if issubclass(c.__class__, RunnerCallback)]

    def base_run_play(
        self,
        trainer_only: bool,
        parameter: Optional[RLParameter],
        memory: Optional[RLMemory],
        trainer: Optional[RLTrainer],
        workers: Optional[List[WorkerRun]],
        callbacks: List[CallbackType],
    ) -> base_run.RunState:
        # --- make instance ---
        if parameter is None:
            parameter = self.make_parameter()
        if memory is None:
            memory = self.make_memory()
        # ---------------------

        # --- random ---
        if self.config.seed is not None:
            common.set_seed(self.config.seed, self.config.seed_enable_gpu)
            self.context.seed = self.config.seed
        # --------------

        # --- callback
        _callbacks = callbacks[:]
        for c in _callbacks:
            if issubclass(c.__class__, RunnerCallback):
                cast(RunnerCallback, c).runner = self

        # --- play ----
        if not trainer_only:
            if not self.context.disable_trainer and trainer is None:
                trainer = self.make_trainer(parameter, memory)
            if workers is None:
                workers = self.make_workers(parameter, memory)
            state = base_run.play(
                self.context,
                self.make_env(),
                parameter,
                memory,
                workers,
                trainer,
                cast(List[RunCallback], [c for c in _callbacks if issubclass(c.__class__, RunCallback)]),
            )
        else:
            if trainer is None:
                trainer = self.make_trainer(parameter, memory)
            state = base_run.play_trainer_only(
                self.context,
                trainer,
                cast(List[TrainerCallback], [c for c in _callbacks if issubclass(c.__class__, TrainerCallback)]),
            )
        # ----------------

        # --- callback
        for c in _callbacks:
            if issubclass(c.__class__, RunnerCallback):
                cast(RunnerCallback, c).runner = None

        return state
