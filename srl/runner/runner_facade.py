from dataclasses import dataclass
from typing import Any, List, Optional, Union

from srl.base.define import RenderModes
from srl.base.rl.base import RLMemory, RLParameter, RLTrainer
from srl.base.run.context import RLWorkerType, RunNameTypes, StrWorkerType
from srl.runner.callback import CallbackType
from srl.runner.runner import Runner


@dataclass()
class RunnerFacade(Runner):
    def get_history(self):
        assert self.history_viewer is not None
        return self.history_viewer

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
        eval_players: List[Union[None, StrWorkerType, RLWorkerType]] = [],
        eval_shuffle_player: bool = False,
        eval_used_device_tf: str = "/CPU",
        eval_used_device_torch: str = "cpu",
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
        """
        self.context_controller.setup(self.config.training_mode, self.config.wkdir)

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
        eval_players: List[Union[None, StrWorkerType, RLWorkerType]] = [],
        eval_shuffle_player: bool = False,
        eval_used_device_tf: str = "/CPU",
        eval_used_device_torch: str = "cpu",
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
        self.context_controller.setup(self.config.training_mode, self.config.wkdir)

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
        )

    def train(
        self,
        # --- stop config
        max_episodes: int = -1,
        timeout: int = -1,
        max_steps: int = -1,
        max_train_count: int = -1,
        max_memory: int = -1,
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
        eval_players: List[Union[None, StrWorkerType, RLWorkerType]] = [],
        eval_shuffle_player: bool = False,
        eval_enable_tf_device: bool = True,
        eval_used_device_tf: str = "/CPU",
        eval_used_device_torch: str = "cpu",
        # --- other
        callbacks: List[CallbackType] = [],
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
        trainer: Optional[RLTrainer] = None,
    ):
        """train

        Args:
            max_episodes (int, optional): 終了するまでのエピソード数. Defaults to -1.
            timeout (int, optional): 終了するまでの時間（秒）. Defaults to -1.
            max_steps (int, optional): 終了するまでの総ステップ. Defaults to -1.
            max_train_count (int, optional): 終了するまでの学習回数. Defaults to -1.
            max_memory (int, optional): 終了するまでのメモリ数. Defaults to -1.
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
            callbacks (List[CallbackType], optional): callbacks. Defaults to [].
        """

        self.context.callbacks = callbacks[:]  # type: ignore , type ok

        # --- set context
        self.context.run_name = RunNameTypes.main
        self.context.distributed = False
        # stop config
        self.context.max_episodes = max_episodes
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = max_train_count
        self.context.max_memory = max_memory
        # play config
        self.context.shuffle_player = shuffle_player
        self.context.disable_trainer = disable_trainer
        # play info
        self.context.training = True
        self.context.render_mode = RenderModes.none

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
                    eval_enable_tf_device=eval_enable_tf_device,
                    eval_used_device_tf=eval_used_device_tf,
                    eval_used_device_torch=eval_used_device_torch,
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

        self.core_play(
            trainer_only=False,
            parameter=parameter,
            memory=memory,
            trainer=trainer,
        )

        # --- history ---
        if self._history_on_memory_callback is not None:
            from srl.runner.callbacks.history_viewer import HistoryViewer

            self.history_viewer = HistoryViewer()
            self.history_viewer.set_history_on_memory(self)
        elif self._history_on_file_callback is not None:
            from srl.runner.callbacks.history_viewer import HistoryViewer

            self.history_viewer = HistoryViewer()
            self.history_viewer.load(self.context.wkdir)
        # ----------------

    def rollout(
        self,
        # --- stop config
        max_episodes: int = -1,
        timeout: int = -1,
        max_steps: int = -1,
        max_memory: int = -1,
        # --- play config
        shuffle_player: bool = True,
        # --- progress
        enable_progress: bool = True,
        progress_start_time: int = 1,
        progress_interval_limit: int = 60 * 10,
        progress_env_info: bool = False,
        progress_train_info: bool = True,
        progress_worker_info: bool = True,
        progress_worker: int = 0,
        # --- other
        callbacks: List[CallbackType] = [],
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
    ):
        """collect_memory"""

        self.context.callbacks = callbacks[:]  # type: ignore , type ok

        # --- set context
        self.context.run_name = RunNameTypes.main
        self.context.distributed = False
        # stop config
        self.context.max_episodes = max_episodes
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_memory = max_memory
        # play config
        self.context.shuffle_player = shuffle_player
        self.context.disable_trainer = True
        # play info
        self.context.training = True
        self.context.render_mode = RenderModes.none

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
                    enable_eval=False,
                )
            )
        # ----------------

        # --- history ---
        if self._history_on_memory_callback is not None:
            self.context.callbacks.append(self._history_on_memory_callback)
        if self._history_on_file_callback is not None:
            self.context.callbacks.append(self._history_on_file_callback)
        # ----------------

        self.core_play(trainer_only=False, parameter=parameter, memory=memory)

        # --- history ---
        if self._history_on_memory_callback is not None:
            from srl.runner.callbacks.history_viewer import HistoryViewer

            self.history_viewer = HistoryViewer()
            self.history_viewer.set_history_on_memory(self)
        elif self._history_on_file_callback is not None:
            from srl.runner.callbacks.history_viewer import HistoryViewer

            self.history_viewer = HistoryViewer()
            self.history_viewer.load(self.context.wkdir)
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
        eval_players: List[Union[None, StrWorkerType, RLWorkerType]] = [],
        eval_shuffle_player: bool = False,
        eval_enable_tf_device: bool = True,
        eval_used_device_tf: str = "/CPU",
        eval_used_device_torch: str = "cpu",
        # --- other
        callbacks: List[CallbackType] = [],
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
        trainer: Optional[RLTrainer] = None,
    ):
        """Trainerが学習するだけでWorkerによるシミュレーションはありません。"""

        self.context.callbacks = callbacks[:]  # type: ignore , type ok

        # --- set context
        self.context.run_name = RunNameTypes.main
        self.context.distributed = False
        # stop config
        self.context.timeout = timeout
        self.context.max_train_count = max_train_count
        # play info
        self.context.training = True
        self.context.render_mode = RenderModes.none

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
                    eval_enable_tf_device=eval_enable_tf_device,
                    eval_used_device_tf=eval_used_device_tf,
                    eval_used_device_torch=eval_used_device_torch,
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

        self.core_play(
            trainer_only=True,
            parameter=parameter,
            memory=memory,
            trainer=trainer,
        )

        # --- history ---
        if self._history_on_memory_callback is not None:
            from srl.runner.callbacks.history_viewer import HistoryViewer

            self.history_viewer = HistoryViewer()
            self.history_viewer.set_history_on_memory(self)
        elif self._history_on_file_callback is not None:
            from srl.runner.callbacks.history_viewer import HistoryViewer

            self.history_viewer = HistoryViewer()
            self.history_viewer.load(self.context.wkdir)
        # ----------------

    def train_mp(
        self,
        # mp
        actor_num: int = 1,
        trainer_parameter_send_interval_by_train_count: int = 100,
        actor_parameter_sync_interval_by_step: int = 100,
        enable_prepare_batch: bool = False,
        # --- stop config
        max_episodes: int = -1,
        timeout: int = -1,
        max_steps: int = -1,
        max_train_count: int = -1,
        max_memory: int = -1,
        # --- play config
        shuffle_player: bool = True,
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
        eval_players: List[Union[None, StrWorkerType, RLWorkerType]] = [],
        eval_shuffle_player: bool = False,
        eval_enable_tf_device: bool = True,
        eval_used_device_tf: str = "/CPU",
        eval_used_device_torch: str = "cpu",
        # --- other
        callbacks: List[CallbackType] = [],
    ):
        """multiprocessingを使用した分散学習による学習を実施します。"""

        self.context.callbacks = callbacks[:]  # type: ignore , type ok

        self.context.actor_num = actor_num
        self.context.actor_parameter_sync_interval_by_step = actor_parameter_sync_interval_by_step
        self.context.trainer_parameter_send_interval_by_train_count = trainer_parameter_send_interval_by_train_count
        self.context.enable_prepare_batch = enable_prepare_batch

        # --- set context
        self.context.run_name = RunNameTypes.main
        self.context.distributed = True
        # stop config
        self.context.max_episodes = max_episodes
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = max_train_count
        self.context.max_memory = max_memory
        # play config
        self.context.shuffle_player = shuffle_player
        # play info
        self.context.training = True
        self.context.render_mode = RenderModes.none

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
                    eval_enable_tf_device=eval_enable_tf_device,
                    eval_used_device_tf=eval_used_device_tf,
                    eval_used_device_torch=eval_used_device_torch,
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

        train(self)

        # --- history ---
        if self._history_on_file_callback is not None:
            from srl.runner.callbacks.history_viewer import HistoryViewer

            self.history_viewer = HistoryViewer()
            self.history_viewer.load(self.context.wkdir)
        # ----------------

    def train_mp_debug(
        self,
        # mp
        actor_num: int = 1,
        trainer_parameter_send_interval_by_train_count: int = 100,
        actor_parameter_sync_interval_by_step: int = 100,
        enable_prepare_batch: bool = False,
        # --- stop config
        max_episodes: int = -1,
        timeout: int = -1,
        max_steps: int = -1,
        max_train_count: int = -1,
        max_memory: int = -1,
        # --- play config
        shuffle_player: bool = True,
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
        eval_players: List[Union[None, StrWorkerType, RLWorkerType]] = [],
        eval_shuffle_player: bool = False,
        eval_enable_tf_device: bool = True,
        eval_used_device_tf: str = "/CPU",
        eval_used_device_torch: str = "cpu",
        # --- other
        callbacks: List[CallbackType] = [],
        # --- debug option
        choice_method: str = "random",
    ):
        """multiprocessingの分散学習と出来る限り似た学習を、single processで実施します。
        ほとんどの引数は train_mp と同じなのです。

        Args:
            choice_method(str, optional): 各actorとtrainerの採用方法を指定します. Defaults to 'random'.
        """
        from .core_mp_debug import train

        self.context.callbacks = callbacks[:]  # type: ignore , type ok

        self.context.actor_num = actor_num
        self.context.actor_parameter_sync_interval_by_step = actor_parameter_sync_interval_by_step
        self.context.trainer_parameter_send_interval_by_train_count = trainer_parameter_send_interval_by_train_count
        self.context.enable_prepare_batch = enable_prepare_batch

        # --- set context
        self.context.run_name = RunNameTypes.main
        self.context.distributed = True
        # stop config
        self.context.max_episodes = max_episodes
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = max_train_count
        self.context.max_memory = max_memory
        # play config
        self.context.shuffle_player = shuffle_player
        # play info
        self.context.training = True
        self.context.render_mode = RenderModes.none

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
                    eval_enable_tf_device=eval_enable_tf_device,
                    eval_used_device_tf=eval_used_device_tf,
                    eval_used_device_torch=eval_used_device_torch,
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

        train(self, choice_method)

        # --- history ---
        if self._history_on_file_callback is not None:
            from srl.runner.callbacks.history_viewer import HistoryViewer

            self.history_viewer = HistoryViewer()
            self.history_viewer.load(self.context.wkdir)
        # ----------------

    def train_rabbitmq(
        self,
        host: str,
        port: int = 5672,
        user: str = "guest",
        password: str = "guest",
        actor_num: int = 1,
        trainer_parameter_send_interval_by_train_count: int = 100,
        actor_parameter_sync_interval_by_step: int = 100,
        enable_prepare_batch: bool = False,
        # --- stop config
        max_episodes: int = -1,
        timeout: int = -1,
        max_steps: int = -1,
        max_train_count: int = -1,
        max_memory: int = -1,
        # --- play config
        shuffle_player: bool = True,
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
        eval_players: List[Union[None, StrWorkerType, RLWorkerType]] = [],
        eval_shuffle_player: bool = False,
        eval_enable_tf_device: bool = True,
        eval_used_device_tf: str = "/CPU",
        eval_used_device_torch: str = "cpu",
        # --- other
        callbacks: List[CallbackType] = [],
    ):
        self.context.callbacks = callbacks[:]  # type: ignore , type ok

        self.context.actor_num = actor_num
        self.context.actor_parameter_sync_interval_by_step = actor_parameter_sync_interval_by_step
        self.context.trainer_parameter_send_interval_by_train_count = trainer_parameter_send_interval_by_train_count
        self.context.enable_prepare_batch = enable_prepare_batch

        # --- set context
        self.context.run_name = RunNameTypes.main
        self.context.distributed = True
        # stop config
        self.context.max_episodes = max_episodes
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = max_train_count
        self.context.max_memory = max_memory
        # play config
        self.context.shuffle_player = shuffle_player
        # play info
        self.context.training = True
        self.context.render_mode = RenderModes.none

        self.context_controller.setup(self.config.training_mode, self.config.wkdir)

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
                    eval_enable_tf_device=eval_enable_tf_device,
                    eval_used_device_tf=eval_used_device_tf,
                    eval_used_device_torch=eval_used_device_torch,
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

        from .rabbitmq.client import run

        run(self, host, port, user, password)

        # --- history ---
        if self._history_on_file_callback is not None:
            from srl.runner.callbacks.history_viewer import HistoryViewer

            self.history_viewer = HistoryViewer()
            self.history_viewer.load(self.context.wkdir)
        # ----------------

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

        Returns:
            Union[List[float], List[List[float]]]: プレイヤー数が1人なら Lost[float]、複数なら List[List[float]]] を返します。
        """

        self.context.callbacks = callbacks[:]  # type: ignore , type ok

        # --- set context
        self.context.run_name = RunNameTypes.main
        self.context.distributed = False
        # stop config
        self.context.max_episodes = max_episodes
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        # play config
        self.context.shuffle_player = shuffle_player
        self.context.disable_trainer = True
        # play info
        self.context.training = False
        self.context.render_mode = RenderModes.none

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

        state = self.core_play(trainer_only=False)

        if self.env_config.player_num == 1:
            return [r[0] for r in state.episode_rewards_list]
        else:
            return state.episode_rewards_list

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
    ):
        self.context.callbacks = callbacks[:]  # type: ignore , type ok

        mode = RenderModes.terminal

        # --- set context
        self.context.run_name = RunNameTypes.main
        self.context.distributed = False
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        # play config
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.training = False
        self.context.render_mode = mode

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

        state = self.core_play(trainer_only=False)

        return state.episode_rewards_list[0]

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
    ):
        self.context.callbacks = callbacks[:]  # type: ignore , type ok

        mode = RenderModes.window

        # --- set context
        self.context.run_name = RunNameTypes.main
        self.context.distributed = False
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        # play config
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.training = False
        self.context.render_mode = mode

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

        state = self.core_play(trainer_only=False)

        return state.episode_rewards_list[0]

    def animation_save_gif(
        self,
        path: str,
        # rendering
        render_kwargs: dict = {},
        step_stop: bool = False,
        render_skip_step: bool = True,
        # render option
        render_interval: float = -1,  # ms
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
    ):
        self.context.callbacks = callbacks[:]  # type: ignore , type ok

        mode = RenderModes.rgb_array

        # --- set context
        self.context.run_name = RunNameTypes.main
        self.context.distributed = False
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        # play config
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.training = False
        self.context.render_mode = mode

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

        state = self.core_play(trainer_only=False)

        rendering.save_gif(path, render_interval, draw_info)

        return state.episode_rewards_list[0]

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
    ):
        self.context.callbacks = callbacks[:]  # type: ignore , type ok

        mode = RenderModes.rgb_array

        # --- set context
        self.context.run_name = RunNameTypes.main
        self.context.distributed = False
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        # play config
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.training = False
        self.context.render_mode = mode

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

        state = self.core_play(trainer_only=False)

        rendering.display(render_interval, render_scale, draw_info)

        return state.episode_rewards_list[0]

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
        _is_test: bool = False,  # for test
    ):
        self.context.callbacks = callbacks[:]  # type: ignore , type ok

        mode = RenderModes.rgb_array

        # --- set context
        self.context.run_name = RunNameTypes.main
        self.context.distributed = False
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        # play config
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.training = False
        self.context.render_mode = mode

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

        window = RePlayableGame(self, _is_test)
        window.play()

    def play_terminal(
        self,
        players: List[Union[None, StrWorkerType, RLWorkerType]] = ["human"],
        # Rendering
        render_kwargs: dict = {},
        step_stop: bool = False,
        render_skip_step: bool = True,
        # --- stop config
        timeout: int = -1,
        max_steps: int = -1,
        # --- other
        callbacks: List[CallbackType] = [],
    ):
        self.context.callbacks = callbacks[:]  # type: ignore , type ok
        self.context.players = players

        mode = RenderModes.terminal

        # --- set context
        self.context.run_name = RunNameTypes.main
        self.context.distributed = False
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        # play config
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.training = False
        self.context.render_mode = mode

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

        state = self.core_play(trainer_only=False)

        return state.episode_rewards_list[0]

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
        self.context.callbacks = callbacks[:]  # type: ignore , type ok

        mode = RenderModes.rgb_array

        # --- set context
        self.context.run_name = RunNameTypes.main
        self.context.distributed = False
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        # play config
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.training = enable_remote_memory
        self.context.render_mode = mode

        self.context_controller.setup(self.config.training_mode, self.config.wkdir)

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
