import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, List, Literal, Optional, Union, cast

from srl.base.context import RunContext, RunState
from srl.base.define import PlayersType
from srl.base.env.config import EnvConfig
from srl.base.env.env_run import EnvRun
from srl.base.env.registration import make as make_env
from srl.base.rl.config import DummyRLConfig, TRLConfig
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.rl.registration import make_memory, make_parameter, make_trainer, make_worker, make_workers
from srl.base.rl.trainer import RLTrainer
from srl.base.rl.worker_run import WorkerRun
from srl.utils.common import load_file

if TYPE_CHECKING:
    import psutil

    from srl.runner.callbacks.history_viewer import HistoryViewer, HistoryViewers

logger = logging.getLogger(__name__)


@dataclass
class RunnerBase(Generic[TRLConfig]):
    """実行環境を提供"""

    #: EnvConfigを指定（文字列のみのIDでも可能）
    name_or_env_config: Optional[Union[str, EnvConfig]] = None
    #: RLConfigを指定, Noneの場合、dummyアルゴリズムが使われます
    rl_config: Optional[TRLConfig] = None  # type: ignore , type

    context: Optional[RunContext] = None  # type: ignore , type
    delay_make_env: bool = False

    def __post_init__(self):
        if (self.name_or_env_config is None) and (self.context is None):
            raise ValueError("Specify one of the following: 'name_or_env_config', 'context'")

        if self.name_or_env_config is None:
            assert self.context.env_config is not None
            self.env_config: EnvConfig = self.context.env_config
        elif isinstance(self.name_or_env_config, str):
            self.env_config: EnvConfig = EnvConfig(self.name_or_env_config)
        else:
            self.env_config: EnvConfig = self.name_or_env_config
        if self.rl_config is None:
            if self.context is not None:
                self.rl_config = self.context.rl_config  # type: ignore
            if self.rl_config is None:
                self.rl_config: TRLConfig = cast(TRLConfig, DummyRLConfig())

        if self.context is None:
            self.context: RunContext = RunContext()
        self.context.env_config = self.env_config
        self.context.rl_config = self.rl_config

        self._env: Optional[EnvRun] = None
        self._worker: Optional[WorkerRun] = None
        self._workers: Optional[List["WorkerRun"]] = None
        self._parameter: Optional[RLParameter] = None
        self._memory: Optional[RLMemory] = None
        self._trainer: Optional[RLTrainer] = None

        self._parameter_dat: Optional[Any] = None
        self._memory_dat: Optional[Any] = None
        self.state: Optional[RunState] = None

        self._history_on_file_kwargs: Optional[dict] = None
        self._history_on_memory_kwargs: Optional[dict] = None
        self._callback_history_on_memory = None
        self._callback_history_on_file = None
        self._checkpoint_kwargs: Optional[dict] = None
        self.history_viewer: Optional["HistoryViewer"] = None
        self._mlflow_kwargs: Optional[dict] = None

        self._progress_kwargs: dict = {}
        self.set_progress()

        self._is_setup_psutil: bool = False
        self._psutil_process: Optional["psutil.Process"] = None

        if not self.delay_make_env:
            self.make_env()

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
    def worker(self) -> WorkerRun:
        return self.make_worker()

    @property
    def workers(self) -> List[WorkerRun]:
        return self.make_workers()

    # ------------------------------
    # set config
    # ------------------------------
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
    # save/load
    # ------------------------------
    def save_parameter(self, path: str, compress: bool = True, **kwargs):
        """save parameter

        Args:
            path (str): save path
            compress (bool, optional): 圧縮するかどうか。圧縮はlzma形式です. Defaults to True.
        """
        self.make_parameter().save(path, compress, **kwargs)

    def load_parameter(self, path: str):
        """load parameter"""
        self._parameter_dat = load_file(path)

    def save_memory(self, path: str, compress: bool = True, **kwargs):
        """save memory

        Args:
            path (str): save path
            compress (bool, optional): 圧縮するかどうか。圧縮はlzma形式です. Defaults to True.
        """
        self.make_memory().save(path, compress, **kwargs)

    def load_memory(self, path: str):
        """load memory"""
        self._memory_dat = load_file(path)

    # ------------------------------
    # make functions
    # ------------------------------
    def make_env(self) -> EnvRun:
        if self._env is None:
            self._env = make_env(self.env_config)
            self.rl_config.setup(self._env)
        return self._env

    def make_parameter(self) -> RLParameter:
        self.context.setup_device()
        if self._parameter is None:
            self._parameter = make_parameter(self.rl_config)
        if self._parameter_dat is not None:
            self._parameter.restore(self._parameter_dat)
            self._parameter_dat = None
        return self._parameter

    def make_memory(self) -> RLMemory:
        self.context.setup_device()
        if self._memory is None:
            self._memory = make_memory(self.rl_config)
        if self._memory_dat is not None:
            self._memory.restore(self._memory_dat)
            self._memory_dat = None
        return self._memory

    def make_trainer(
        self,
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
    ) -> RLTrainer:
        self.context.setup_device()
        if self._trainer is None:
            if parameter is None:
                parameter = self.make_parameter()
            if memory is None:
                memory = self.make_memory()
            self._trainer = make_trainer(self.rl_config, parameter, memory)
        if (self._parameter_dat is not None) and (self._parameter is not None):
            self._parameter.restore(self._parameter_dat)
            self._parameter_dat = None
        if (self._memory_dat is not None) and (self._memory is not None):
            self._memory.restore(self._memory_dat)
            self._memory_dat = None
        return self._trainer

    def make_worker(
        self,
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
    ) -> WorkerRun:
        self.context.setup_device()
        if self._worker is None:
            if parameter is None:
                parameter = self.make_parameter()
            if memory is None:
                memory = self.make_memory()
            self._worker = make_worker(self.rl_config, self.make_env(), parameter, memory)
        if (self._parameter_dat is not None) and (self._parameter is not None):
            self._parameter.restore(self._parameter_dat)
            self._parameter_dat = None
        if (self._memory_dat is not None) and (self._memory is not None):
            self._memory.restore(self._memory_dat)
            self._memory_dat = None
        return self._worker

    def make_workers(
        self,
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
    ):
        self.context.setup_device()
        if self._workers is None:
            if parameter is None:
                parameter = self.make_parameter()
            if memory is None:
                memory = self.make_memory()
            self._workers, main_worker_idx = make_workers(
                self.context.players,
                self.make_env(),
                self.rl_config,
                parameter,
                memory,
                self.make_worker(parameter, memory),
            )
        return self._workers

    # ------------------------------
    # device
    # ------------------------------
    def set_device(
        self,
        device: str = "AUTO",
        enable_tf_device: bool = True,
        set_CUDA_VISIBLE_DEVICES_if_CPU: bool = True,
        tf_enable_memory_growth: bool = True,
    ):
        """set device.

        "AUTO",""    : Automatic assignment.
        "CPU","CPU:0": Use CPU.
        "GPU","GPU:0": Use GPU.

        Args:
            device (str, optional): mainのdeviceを指定します。分散学習を用いない場合、これだけが使用されます. Defaults to "AUTO".
            enable_tf_device (bool, optional): tensorflowにて、 'with tf.device()' を使用する. Defaults to True.
            set_CUDA_VISIBLE_DEVICES_if_CPU (bool, optional): CPUの場合 CUDA_VISIBLE_DEVICES を-1にする. Defaults to True.
            tf_enable_memory_growth (bool, optional): tensorflowにて、'set_memory_growth(True)' を実行する. Defaults to True.
        """
        if self.context.is_setup():
            logger.warning("Device cannot be changed after initialization.")
            return
        self.context.device = device
        self.context.enable_tf_device = enable_tf_device
        self.context.set_CUDA_VISIBLE_DEVICES_if_CPU = set_CUDA_VISIBLE_DEVICES_if_CPU
        self.context.tf_enable_memory_growth = tf_enable_memory_growth

    # ------------------------------
    # progress
    # ------------------------------
    def set_progress(
        self,
        start_time: int = 1,
        interval_limit: int = 60 * 2,
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
        eval_players: PlayersType = [],
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

    def apply_progress(self, callbacks: list, apply_eval: bool):
        from srl.runner.callbacks.print_progress import PrintProgress

        if apply_eval:
            callbacks.append(PrintProgress(**self._progress_kwargs))
        else:
            _kwargs = self._progress_kwargs.copy()
            _kwargs["enable_eval"] = False
            callbacks.append(PrintProgress(**_kwargs))
        logger.info("add callback PrintProgress")

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
        interval: Union[float, int] = 1,
        interval_mode: Literal["time", "step"] = "time",
        enable_eval: bool = False,
        eval_episode: int = 1,
        eval_timeout: float = -1,
        eval_max_steps: int = -1,
        eval_players: PlayersType = [],
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
        interval: Union[float, int] = 1,
        interval_mode: Literal["time", "step"] = "time",
        add_history: bool = False,
        write_system: bool = False,
        enable_eval: bool = False,
        eval_episode: int = 1,
        eval_timeout: float = -1,
        eval_max_steps: int = -1,
        eval_players: PlayersType = [],
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

    # ------------------------------
    # checkpoint
    # ------------------------------
    def load_checkpoint(self, save_dir: str):
        from srl.runner.callbacks.checkpoint import Checkpoint

        path = Checkpoint.get_parameter_path(save_dir)
        if os.path.isfile(path):
            try:
                self.make_parameter().load(path)
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
        eval_players: PlayersType = [],
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

    def apply_checkpoint(self, callbacks: list):
        if self._checkpoint_kwargs is None:
            return
        from srl.runner.callbacks.checkpoint import Checkpoint

        callbacks.append(Checkpoint(**self._checkpoint_kwargs))
        logger.info(f"add callback Checkpoint: {self._checkpoint_kwargs['save_dir']}")

    # ------------------------------
    # MLFlow
    # ------------------------------
    def set_mlflow(
        self,
        experiment_name: str = "",
        run_name: str = "",
        tags: dict = {},
        interval: Union[float, int] = 60,
        interval_mode: Literal["time", "step"] = "time",
        eval_interval: float = -1,
        checkpoint_interval: float = 60 * 30,
        enable_checkpoint: bool = True,
        enable_eval: bool = True,
        eval_episode: int = 1,
        eval_timeout: float = -1,
        eval_max_steps: int = -1,
        eval_players: PlayersType = [],
        eval_shuffle_player: bool = False,
    ):
        self._mlflow_kwargs = dict(
            experiment_name=experiment_name,
            run_name=run_name,
            tags=tags,
            interval=interval,
            interval_mode=interval_mode,
            eval_interval=eval_interval,
            checkpoint_interval=checkpoint_interval,
            enable_checkpoint=enable_checkpoint,
            enable_eval=enable_eval,
            eval_episode=eval_episode,
            eval_timeout=eval_timeout,
            eval_max_steps=eval_max_steps,
            eval_players=eval_players,
            eval_shuffle_player=eval_shuffle_player,
        )

    def load_parameter_from_mlflow(
        self,
        experiment_name: str = "",
        run_idx: int = -1,
        parameter_idx: int = -1,
    ):
        from srl.runner.callbacks.mlflow_callback import MLFlowCallback

        if experiment_name == "":
            experiment_name = self.env_config.name

        run_id = MLFlowCallback.get_run_id(experiment_name, self.rl_config.get_name(), run_idx)
        if run_id is None:
            raise ValueError(f"run id is not found. experiment: {experiment_name}, rl_name: {self.rl_config.get_name()}")
        files = MLFlowCallback.get_parameter_files(run_id)
        if len(files) == 0:
            raise ValueError(f"parameter is not found. experiment: {experiment_name}, rl_name: {self.rl_config.get_name()}")
        MLFlowCallback.load_parameter(
            run_id,
            files[parameter_idx],
            self.make_parameter(),
        )

    def make_html_all_parameters_in_mlflow(
        self,
        experiment_name: str = "",
        run_idx: int = -1,
        run_id: Optional[str] = None,
        **render_kwargs,
    ):
        from srl.runner.callbacks.mlflow_callback import MLFlowCallback

        if run_id is None:
            if experiment_name == "":
                experiment_name = self.env_config.name
            run_id = MLFlowCallback.get_run_id(experiment_name, self.rl_config.get_name(), run_idx)
        if run_id is None:
            raise ValueError(f"run id is not found. experiment: {experiment_name}, rl_name: {self.rl_config.get_name()}")
        MLFlowCallback.make_html_all_parameters(run_id, self.env_config, self.rl_config, **render_kwargs)

    def disable_mlflow(self):
        self._mlflow_kwargs = None

    def apply_mlflow(self, callbacks: list):
        if self._mlflow_kwargs is None:
            return
        from srl.runner.callbacks.mlflow_callback import MLFlowCallback

        callbacks.append(MLFlowCallback(**self._mlflow_kwargs))
        logger.info("add callback MLFlowCallback")
