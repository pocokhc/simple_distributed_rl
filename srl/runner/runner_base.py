import datetime
import logging
import os
import pprint
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from srl.base.context import RunContext
from srl.base.define import EnvObservationType, PlayerType, RLObservationType
from srl.base.env.config import EnvConfig
from srl.base.env.env_run import EnvRun
from srl.base.env.registration import make as make_env
from srl.base.rl.config import DummyRLConfig, RLConfig
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.rl.registration import make_memory, make_parameter, make_trainer, make_worker, make_workers
from srl.base.rl.trainer import RLTrainer
from srl.base.rl.worker_run import WorkerRun
from srl.base.run import play
from srl.base.run.callback import CallbackType
from srl.base.run.core_play import RunStateActor
from srl.base.run.core_train_only import RunStateTrainer

if TYPE_CHECKING:
    import psutil

    from srl.runner.callbacks.history_viewer import HistoryViewer, HistoryViewers

logger = logging.getLogger(__name__)


@dataclass
class RunnerBase:
    """実行環境を提供"""

    #: EnvConfigを指定（文字列のみのIDでも可能）
    name_or_env_config: Union[str, EnvConfig]
    #: RLConfigを指定, Noneの場合、dummyアルゴリズムが使われます
    rl_config: Optional[RLConfig] = None  # type: ignore , type

    context: Optional[RunContext] = None  # type: ignore , type

    # --- private(static class instance)
    __setup_process = False

    def __post_init__(self):

        # --- config
        if isinstance(self.name_or_env_config, str):
            self.env_config: EnvConfig = EnvConfig(self.name_or_env_config)
        else:
            self.env_config: EnvConfig = self.name_or_env_config
        if self.rl_config is None:
            self.rl_config: RLConfig = DummyRLConfig()
        if self.context is None:
            self.context: RunContext = RunContext(self.env_config, self.rl_config)

        self._env: Optional[EnvRun] = None
        self._parameter: Optional[RLParameter] = None
        self._memory: Optional[RLMemory] = None
        self._trainer: Optional[RLTrainer] = None
        self._workers: Optional[List[WorkerRun]] = None
        self._main_worker_idx: Optional[int] = None
        self.state: Optional[Union[RunStateActor, RunStateTrainer]] = None

        self._history_on_file_kwargs: Optional[dict] = None
        self._history_on_memory_kwargs: Optional[dict] = None
        self._checkpoint_kwargs: Optional[dict] = None
        self.history_viewer: Optional["HistoryViewer"] = None

        self._progress_kwargs: dict = {}
        self.set_progress_options()

        self._is_setup_psutil: bool = False
        self._psutil_process: Optional["psutil.Process"] = None

    @property
    def env(self) -> EnvRun:
        return self.make_env()

    @property
    def parameter(self) -> RLParameter:
        return self.make_parameter()

    @property
    def memory(self) -> RLMemory:
        return self.make_memory()

    @property
    def trainer(self) -> RLTrainer:
        return self.make_trainer()

    @property
    def workers(self) -> List[WorkerRun]:
        workers, _ = self.make_workers()
        return workers

    # ------------------------------
    # set config
    # ------------------------------
    def set_players(self, players: List[PlayerType] = []):
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
        self.context.seed = seed
        self.context.seed_enable_gpu = seed_enable_gpu

    def enable_stats(self):
        """ハードウェアの統計情報に関する設定を有効にします。"""
        self.context.enable_stats = True

    def disable_stats(self):
        """ハードウェアの統計情報に関する設定を無効にします。"""
        self.context.enable_stats = False

    # ------------------------------
    # model summary
    # ------------------------------
    def model_summary(self, expand_nested: bool = True, **kwargs) -> RLParameter:
        """modelの概要を返します。これは以下と同じです。

        >>> parameter = runner.make_parameter()
        >>> parameter.summary()

        Args:
            expand_nested (bool): tensorflow option

        Returns:
            RLParameter: RLParameter
        """
        parameter = self.make_parameter()
        parameter.summary(expand_nested=expand_nested, **kwargs)
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
    def make_env(self) -> EnvRun:
        if self._env is None:
            self._env = make_env(self.env_config)
            logger.info(f"make env: {self._env.name}")
        return self._env

    def make_parameter(self, is_load: bool = True) -> RLParameter:
        self._setup_process()
        if self._parameter is None:
            if not self.rl_config.is_setup(self.env_config.name):
                self.rl_config.setup(self.make_env())
            self._parameter = make_parameter(self.rl_config, is_load=is_load)
            logger.info(f"make parameter: {self._parameter}")
        return self._parameter

    def make_memory(self, is_load: bool = True) -> RLMemory:
        self._setup_process()
        if self._memory is None:
            if not self.rl_config.is_setup(self.env_config.name):
                self.rl_config.setup(self.make_env())
            self._memory = make_memory(self.rl_config, is_load=is_load)
            logger.info(f"make memory: {self._memory}")
        return self._memory

    def make_trainer(
        self,
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
    ) -> RLTrainer:
        self._setup_process()
        if self._trainer is None:
            if parameter is None:
                parameter = self.make_parameter()
            if memory is None:
                memory = self.make_memory()
            if not self.rl_config.is_setup(self.env_config.name):
                self.rl_config.setup(self.make_env())
            self._trainer = make_trainer(self.rl_config, parameter, memory)
        return self._trainer

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
        return make_worker(self.rl_config, env, parameter, memory)

    def make_workers(
        self,
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
    ) -> Tuple[List[WorkerRun], int]:
        if self._workers is None:
            if parameter is None:
                parameter = self.make_parameter()
            if memory is None:
                memory = self.make_memory()
            self._workers, self._main_worker_idx = make_workers(
                self.context.players,
                self.make_env(),
                self.rl_config,
                parameter,
                memory,
            )
        assert self._main_worker_idx is not None
        return self._workers, self._main_worker_idx

    # ------------------------------
    # process
    # ------------------------------
    def _setup_process(self):
        if RunnerBase.__setup_process:
            return
        self.context.set_memory_limit()
        self.context.set_device()
        RunnerBase.__setup_process = True

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
        if RunnerBase.__setup_process:
            logger.warning("Device cannot be changed after initialization.")
            return

        self.context.device = device
        self.context.set_CUDA_VISIBLE_DEVICES_if_CPU = set_CUDA_VISIBLE_DEVICES_if_CPU
        self.context.tf_device_enable = tf_device_enable
        self.context.tf_enable_memory_growth = tf_enable_memory_growth

    # ------------------------------
    # utility
    # ------------------------------
    def get_env_init_state(self, encode: bool = True) -> Union[EnvObservationType, RLObservationType]:
        env = self.make_env()
        env.setup()
        env.reset()
        state = env.state
        if encode:
            worker = self.make_worker()
            state = worker.state_encode(
                state,
                env,
                create_env_sate=True,
                enable_state_encode=True,
                append_recent_state=False,
                is_dummy=False,
            )
        return state

    def copy(self, env_share: bool) -> "RunnerBase":
        runner = RunnerBase(
            self.env_config.copy(),
            self.rl_config.copy(),
            self.context.copy(),
        )
        if env_share:
            runner._env = self._env
        return runner

    def print_config(self):
        print(f"env\n{pprint.pformat(self.env_config.to_dict())}")
        print(f"rl\n{pprint.pformat(self.rl_config.to_dict())}")
        print(f"context\n{pprint.pformat(self.context.to_dict())}")

    # ------------------------------
    # path
    # ------------------------------
    def get_dirname1(self) -> str:
        dir_name = f"{self.env_config.name}_{self.rl_config.get_name()}"
        dir_name = re.sub(r'[\\/:?."<>\|]', "_", dir_name)
        return dir_name

    def get_dirname2(self) -> str:
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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
        enable_eval: bool = False,
        eval_shuffle_player: bool = False,
        eval_episode: int = 1,
        eval_timeout: float = -1,
        eval_max_steps: int = -1,
        eval_players: List[PlayerType] = [],
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
            eval_shuffle_player (bool, optional): 評価時にplayersをシャッフルするか. Defaults to False.
            eval_episode (int, optional): 評価時のエピソード数. Defaults to 1.
            eval_timeout (int, optional): 評価時の1エピソードの制限時間. Defaults to -1.
            eval_max_steps (int, optional): 評価時の1エピソードの最大ステップ数. Defaults to -1.
            eval_players (List[Union[None, str, Tuple[str, dict], RLConfig]], optional): 評価時のplayers. Defaults to [].
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
            eval_shuffle_player=eval_shuffle_player,
            enable_eval=enable_eval,
            eval_episode=eval_episode,
            eval_timeout=eval_timeout,
            eval_max_steps=eval_max_steps,
            eval_players=eval_players,
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
        assert self.history_viewer is not None
        return self.history_viewer

    def set_history_on_memory(
        self,
        interval: int = 1,
        interval_mode: str = "time",
        enable_eval: bool = False,
        eval_episode: int = 1,
        eval_timeout: float = -1,
        eval_max_steps: int = -1,
        eval_players: List[PlayerType] = [],
        eval_shuffle_player: bool = False,
    ):
        """学習履歴を保存する設定を指定します。

        Args:
            interval (int, optional): 学習履歴を保存する間隔. Defaults to 1.
            interval_mode (str, optional): 学習履歴を保存する間隔の単位(time:秒、step:step). Defaults to "time".
            enable_eval (bool, optional): 学習履歴の保存時に評価用のシミュレーションを実行します. Defaults to False.
            eval_episode (int, optional): 評価時のエピソード数. Defaults to 1.
            eval_timeout (int, optional): 評価時の1エピソードの制限時間. Defaults to -1.
            eval_max_steps (int, optional): 評価時の1エピソードの最大ステップ数. Defaults to -1.
            eval_players (List[Union[None, str, Tuple[str, dict], RLConfig]], optional): 評価時のplayers. Defaults to [].
            eval_shuffle_player (bool, optional): 評価時にplayersをシャッフルするか. Defaults to False.
        """

        self._history_on_memory_kwargs = dict(
            interval=interval,
            interval_mode=interval_mode,
            enable_eval=enable_eval,
            eval_episode=eval_episode,
            eval_timeout=eval_timeout,
            eval_max_steps=eval_max_steps,
            eval_players=eval_players,
            eval_shuffle_player=eval_shuffle_player,
        )

    def set_history_on_file(
        self,
        save_dir: str = "",
        interval: int = 1,
        interval_mode: str = "time",
        add_history: bool = False,
        write_system: bool = False,
        enable_eval: bool = False,
        eval_episode: int = 1,
        eval_timeout: float = -1,
        eval_max_steps: int = -1,
        eval_players: List[PlayerType] = [],
        eval_shuffle_player: bool = False,
    ):
        """学習履歴を保存する設定を指定します。

        Args:
            save_dir (str, optional): 保存するディレクトリ、""の場合tmpフォルダを作成
            interval (int, optional): 学習履歴を保存する間隔. Defaults to 1.
            interval_mode (str, optional): 学習履歴を保存する間隔の単位(time:秒、step:step). Defaults to "time".
            add_history (bool, optional): 追記で学習履歴を保存. Defaults to False.
            write_system (bool, optional): CPU/memory情報も保存. Defaults to False.
            enable_eval (bool, optional): 学習履歴の保存時に評価用のシミュレーションを実行します. Defaults to False.
            eval_episode (int, optional): 評価時のエピソード数. Defaults to 1.
            eval_timeout (int, optional): 評価時の1エピソードの制限時間. Defaults to -1.
            eval_max_steps (int, optional): 評価時の1エピソードの最大ステップ数. Defaults to -1.
            eval_players (List[Union[None, str, Tuple[str, dict], RLConfig]], optional): 評価時のplayers. Defaults to [].
            eval_shuffle_player (bool, optional): 評価時にplayersをシャッフルするか. Defaults to False.
        """

        self._history_on_file_kwargs = dict(
            save_dir=save_dir,
            interval=interval,
            interval_mode=interval_mode,
            add_history=add_history,
            write_system=write_system,
            enable_eval=enable_eval,
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
        is_load: bool,
        interval: int = 60 * 10,
        enable_eval: bool = True,
        eval_episode: int = 1,
        eval_timeout: float = -1,
        eval_max_steps: int = -1,
        eval_players: List[PlayerType] = [],
        eval_shuffle_player: bool = False,
    ):
        """一定間隔でモデルを保存します。

        Args:
            save_dir (int): 保存するディレクトリ
            interval (int, optional): 保存する間隔（秒）. Defaults to 60*10sec.
            enable_eval (bool, optional): モデル保存時に評価用のシミュレーションを実行します. Defaults to False.
            eval_episode (int, optional): 評価時のエピソード数. Defaults to 1.
            eval_timeout (int, optional): 評価時の1エピソードの制限時間. Defaults to -1.
            eval_max_steps (int, optional): 評価時の1エピソードの最大ステップ数. Defaults to -1.
            eval_players (List[Union[None, str, Tuple[str, dict], RLConfig]], optional): 評価時のplayers. Defaults to [].
            eval_shuffle_player (bool, optional): 評価時にplayersをシャッフルするか. Defaults to False.
        """
        if is_load:
            self.load_checkpoint(save_dir)

        self._checkpoint_kwargs = dict(
            save_dir=save_dir,
            interval=interval,
            enable_eval=enable_eval,
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
    def apply_progress(self, callbacks: list, enable_eval: bool):
        from srl.runner.callbacks.print_progress import PrintProgress

        if enable_eval:
            callbacks.append(PrintProgress(**self._progress_kwargs))
        else:
            _kwargs = self._progress_kwargs.copy()
            _kwargs["enable_eval"] = False
            callbacks.append(PrintProgress(**_kwargs))
        logger.info("add callback PrintProgress")

    def apply_checkpoint(self, callbacks: list):
        if self._checkpoint_kwargs is None:
            return
        from srl.runner.callbacks.checkpoint import Checkpoint

        callbacks.append(Checkpoint(**self._checkpoint_kwargs))
        logger.info(f"add callback Checkpoint: {self._checkpoint_kwargs['save_dir']}")

    def _apply_history(self, callbacks: list):
        self._callback_history_on_memory = None
        self._callback_history_on_file = None
        if self._history_on_memory_kwargs is not None:
            if self.context.distributed:
                logger.info("HistoryOnMemory is disabled.")
            else:
                from srl.runner.callbacks.history_on_memory import HistoryOnMemory

                self._callback_history_on_memory = HistoryOnMemory(**self._history_on_memory_kwargs)
                callbacks.append(self._callback_history_on_memory)
                logger.info("add callback HistoryOnMemory")

        if self._history_on_file_kwargs is not None:
            from srl.runner.callbacks.history_on_file import HistoryOnFile

            self._callback_history_on_file = HistoryOnFile(**self._history_on_file_kwargs)
            callbacks.append(self._callback_history_on_file)
            logger.info(f"add callback HistoryOnFile: {self._history_on_file_kwargs['save_dir']}")

    def _after_history(self):
        if self._callback_history_on_memory is not None:
            from srl.runner.callbacks.history_viewer import HistoryViewer

            self.history_viewer = HistoryViewer()
            self.history_viewer.set_history_on_memory(self._callback_history_on_memory, self)
        elif self._callback_history_on_file is not None:
            from srl.runner.callbacks.history_viewer import HistoryViewer

            self.history_viewer = HistoryViewer()
            self.history_viewer.load(self._callback_history_on_file._base.save_dir)

            # 2回目以降は引き継ぐ
            if self._history_on_file_kwargs is not None:
                self._history_on_file_kwargs["add_history"] = True

    def run_context(
        self,
        reset_workers: bool = True,
        reset_trainer: bool = True,
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
        trainer: Optional[RLTrainer] = None,
        workers: Optional[List[WorkerRun]] = None,
        main_worker_idx: int = 0,
        callbacks: List[CallbackType] = [],
        logger_config: bool = False,
    ):
        if reset_workers:
            self._workers = None
            self._main_worker_idx = None
        if reset_trainer:
            self._trainer = None

        # --- make instance
        if workers is None:
            if parameter is None:
                parameter = self.make_parameter()
            if memory is None:
                memory = self.make_memory()
            workers, main_worker_idx = self.make_workers(parameter, memory)
        if not self.context.disable_trainer and trainer is None:
            if parameter is None:
                parameter = self.make_parameter()
            if memory is None:
                memory = self.make_memory()
            trainer = self.make_trainer(parameter, memory)

        self.state = play.play(
            self.context,
            self.make_env(),
            workers,
            main_worker_idx,
            trainer,
            callbacks,
            logger_config=logger_config,
        )
        return self.state

    def run_context_generator(
        self,
        reset_workers: bool = True,
        reset_trainer: bool = True,
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
        trainer: Optional[RLTrainer] = None,
        workers: Optional[List[WorkerRun]] = None,
        main_worker_idx: int = 0,
        callbacks: List[CallbackType] = [],
        logger_config: bool = False,
    ):
        if reset_workers:
            self._workers = None
            self._main_worker_idx = None
        if reset_trainer:
            self._trainer = None

        # --- make instance
        if workers is None:
            if parameter is None:
                parameter = self.make_parameter()
            if memory is None:
                memory = self.make_memory()
            workers, main_worker_idx = self.make_workers(parameter, memory)
        if not self.context.disable_trainer and trainer is None:
            if parameter is None:
                parameter = self.make_parameter()
            if memory is None:
                memory = self.make_memory()
            trainer = self.make_trainer(parameter, memory)

        return play.play_generator(
            self.context,
            self.make_env(),
            workers,
            main_worker_idx,
            trainer,
            callbacks,
            logger_config=logger_config,
        )

    def run_context_trainer_only(
        self,
        reset_trainer: bool = True,
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
        trainer: Optional[RLTrainer] = None,
        callbacks: List[CallbackType] = [],
        logger_config: bool = False,
    ):
        if reset_trainer:
            self._trainer = None

        # --- make instance
        if trainer is None:
            if parameter is None:
                parameter = self.make_parameter()
            if memory is None:
                memory = self.make_memory()
            trainer = self.make_trainer(parameter, memory)

        self.state = play.play_trainer_only(
            self.context,
            trainer,
            callbacks,
            logger_config=logger_config,
        )
        return self.state
