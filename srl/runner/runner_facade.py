import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Union, cast

from srl.base.define import RenderModes
from srl.base.rl.base import RLMemory, RLParameter, RLTrainer
from srl.base.run.context import RLWorkerType, RunNameTypes, StrWorkerType
from srl.runner.runner import CallbackType, Runner

if TYPE_CHECKING:
    from srl.runner.distribution.callback import DistributionCallback
    from srl.runner.distribution.connectors.redis_ import RedisParameters

logger = logging.getLogger(__name__)


@dataclass()
class RunnerFacade(Runner):
    def train(
        self,
        # --- stop config
        max_episodes: int = -1,
        timeout: float = -1,
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
        # --- eval
        enable_eval: bool = False,
        eval_env_sharing: bool = False,
        eval_episode: int = 1,
        eval_timeout: float = -1,
        eval_max_steps: int = -1,
        eval_players: List[Union[None, StrWorkerType, RLWorkerType]] = [],
        eval_shuffle_player: bool = False,
        # --- other
        callbacks: List[CallbackType] = [],
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
        trainer: Optional[RLTrainer] = None,
    ):
        """train

        Args:
            max_episodes (int, optional): 終了するまでのエピソード数. Defaults to -1.
            timeout (float, optional): 終了するまでの時間（秒）. Defaults to -1.
            max_steps (int, optional): 終了するまでの総ステップ. Defaults to -1.
            max_train_count (int, optional): 終了するまでの学習回数. Defaults to -1.
            max_memory (int, optional): 終了するまでのメモリ数. Defaults to -1.
            shuffle_player (bool, optional): playersをシャッフルするかどうか. Defaults to True.
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
        callbacks = callbacks[:]

        # --- context
        self.context.run_name = RunNameTypes.main
        # stop config
        self.context.max_episodes = max_episodes
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = max_train_count
        self.context.max_memory = max_memory
        # play config
        self.context.shuffle_player = shuffle_player
        self.context.disable_trainer = False
        # play info
        self.context.distributed = False
        self.context.training = True
        self.context.render_mode = RenderModes.none

        # --- progress ---
        if enable_progress:
            from srl.runner.callbacks.print_progress import PrintProgress

            callbacks.append(
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
                )
            )
            logger.info("add callback PrintProgress")
        # ----------------

        self._base_run_play_before(
            enable_checkpoint=True,
            enable_checkpoint_load=True,
            enable_history_on_memory=True,
            enable_history_on_file=True,
            callbacks=callbacks,
        )
        state = self.base_run_play(
            trainer_only=False,
            parameter=parameter,
            memory=memory,
            trainer=trainer,
            workers=None,
        )
        self._base_run_play_after()
        return state

    def rollout(
        self,
        # --- stop config
        max_episodes: int = -1,
        timeout: float = -1,
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
        callbacks = callbacks[:]

        # --- set context
        self.context.run_name = RunNameTypes.main
        # stop config
        self.context.max_episodes = max_episodes
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = max_memory
        # play config
        self.context.shuffle_player = shuffle_player
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = True
        self.context.render_mode = RenderModes.none

        # --- progress ---
        if enable_progress:
            from srl.runner.callbacks.print_progress import PrintProgress

            callbacks.append(
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
            logger.info("add callback PrintProgress")
        # ----------------

        self._base_run_play_before(
            enable_checkpoint_load=True,
            enable_checkpoint=True,
            enable_history_on_memory=True,
            enable_history_on_file=True,
            callbacks=callbacks,
        )
        state = self.base_run_play(
            trainer_only=False,
            parameter=parameter,
            memory=memory,
            trainer=None,
            workers=None,
        )
        self._base_run_play_after()
        return state

    def train_only(
        self,
        # --- stop config
        timeout: float = -1,
        max_train_count: int = -1,
        # --- progress
        enable_progress: bool = True,
        progress_start_time: int = 1,
        progress_interval_limit: int = 60 * 10,
        progress_train_info: bool = True,
        # --- eval
        enable_eval: bool = False,
        eval_episode: int = 1,
        eval_timeout: float = -1,
        eval_max_steps: int = -1,
        eval_players: List[Union[None, StrWorkerType, RLWorkerType]] = [],
        eval_shuffle_player: bool = False,
        # --- other
        callbacks: List[CallbackType] = [],
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
        trainer: Optional[RLTrainer] = None,
    ):
        """Trainerが学習するだけでWorkerによるシミュレーションはありません。"""
        callbacks = callbacks[:]

        # --- context
        self.context.run_name = RunNameTypes.main
        # stop config
        self.context.max_episodes = 0
        self.context.timeout = timeout
        self.context.max_steps = 0
        self.context.max_train_count = max_train_count
        self.context.max_memory = 0
        # play config
        self.context.shuffle_player = False
        self.context.disable_trainer = False
        # play info
        self.context.distributed = False
        self.context.training = True
        self.context.render_mode = RenderModes.none

        # --- progress ---
        if enable_progress:
            from srl.runner.callbacks.print_progress import PrintProgress

            callbacks.append(
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
                )
            )
            logger.info("add callback PrintProgress")
        # ----------------

        self._base_run_play_before(
            enable_checkpoint_load=True,
            enable_checkpoint=True,
            enable_history_on_memory=True,
            enable_history_on_file=True,
            callbacks=callbacks,
        )
        state = self.base_run_play(
            trainer_only=True,
            parameter=parameter,
            memory=memory,
            trainer=trainer,
            workers=None,
        )
        self._base_run_play_after()
        return state

    def train_mp(
        self,
        # mp
        actor_num: int = 1,
        queue_capacity: int = 1000,
        trainer_parameter_send_interval: int = 1,
        actor_parameter_sync_interval: int = 1,
        enable_prepare_sample_batch: bool = False,
        device_actors: Union[str, List[str]] = "AUTO",
        # --- stop config
        max_episodes: int = -1,
        timeout: float = -1,
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
        eval_timeout: float = -1,
        eval_max_steps: int = -1,
        eval_players: List[Union[None, StrWorkerType, RLWorkerType]] = [],
        eval_shuffle_player: bool = False,
        # --- other
        callbacks: List[CallbackType] = [],
    ):
        """multiprocessingを使用した分散学習による学習を実施します。"""
        callbacks = callbacks[:]

        self.context.actor_num = actor_num
        self.config.dist_queue_capacity = queue_capacity
        self.config.trainer_parameter_send_interval = trainer_parameter_send_interval
        self.config.actor_parameter_sync_interval = actor_parameter_sync_interval
        self.config.dist_enable_prepare_sample_batch = enable_prepare_sample_batch
        self.config.device_actors = device_actors

        # --- set context
        self.context.run_name = RunNameTypes.main
        # stop config
        self.context.max_episodes = max_episodes
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = max_train_count
        self.context.max_memory = max_memory
        # play config
        self.context.shuffle_player = shuffle_player
        self.context.disable_trainer = False
        # play info
        self.context.distributed = True
        self.context.training = True
        self.context.render_mode = RenderModes.none

        if enable_progress:
            from srl.runner.callbacks.print_progress import PrintProgress

            callbacks.append(
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
                )
            )
            logger.info("add callback PrintProgress")

        self._base_run_play_before(
            enable_checkpoint_load=True,
            enable_checkpoint=True,
            enable_history_on_memory=False,
            enable_history_on_file=True,
            callbacks=callbacks,
        )

        from .core_mp import train

        train(self)

        self._base_run_play_after()

    # def train_mp_debug(
    #     self,
    #     # mp
    #     actor_num: int = 1,
    #     queue_capacity: int = 1000,
    #     trainer_parameter_send_interval: int = 1,
    #     actor_parameter_sync_interval: int = 1,
    #     enable_prepare_sample_batch: bool = False,
    #     device_actors: Union[str, List[str]] = "AUTO",
    #     # --- stop config
    #     max_episodes: int = -1,
    #     timeout: float = -1,
    #     max_steps: int = -1,
    #     max_train_count: int = -1,
    #     max_memory: int = -1,
    #     # --- play config
    #     shuffle_player: bool = True,
    #     # --- progress
    #     enable_progress: bool = True,
    #     progress_start_time: int = 1,
    #     progress_interval_limit: int = 60 * 10,
    #     progress_env_info: bool = False,
    #     progress_train_info: bool = True,
    #     progress_worker_info: bool = True,
    #     progress_worker: int = 0,
    #     progress_max_actor: int = 5,
    #     # --- eval
    #     enable_eval: bool = False,
    #     eval_env_sharing: bool = False,
    #     eval_episode: int = 1,
    #     eval_timeout: float = -1,
    #     eval_max_steps: int = -1,
    #     eval_players: List[Union[None, StrWorkerType, RLWorkerType]] = [],
    #     eval_shuffle_player: bool = False,
    #     # --- other
    #     callbacks: List[CallbackType] = [],
    #     # --- debug option
    #     choice_method: str = "random",
    # ):
    #     """multiprocessingの分散学習と出来る限り似た学習を、single processで実施します。
    #     ほとんどの引数は train_mp と同じなのです。

    #     Args:
    #         choice_method(str, optional): 各actorとtrainerの採用方法を指定します. Defaults to 'random'.
    #     """
    #     callbacks = callbacks[:]

    #     self.context.actor_num = actor_num
    #     self.config.dist_queue_capacity = queue_capacity
    #     self.config.trainer_parameter_send_interval = trainer_parameter_send_interval
    #     self.config.actor_parameter_sync_interval = actor_parameter_sync_interval
    #     self.config.dist_enable_prepare_sample_batch = enable_prepare_sample_batch
    #     self.config.device_actors = device_actors

    #     # --- set context
    #     self.context.run_name = RunNameTypes.main
    #     # stop config
    #     self.context.max_episodes = max_episodes
    #     self.context.timeout = timeout
    #     self.context.max_steps = max_steps
    #     self.context.max_train_count = max_train_count
    #     self.context.max_memory = max_memory
    #     # play config
    #     self.context.shuffle_player = shuffle_player
    #     self.context.disable_trainer = False
    #     # play info
    #     self.context.distributed = True
    #     self.context.training = True
    #     self.context.render_mode = RenderModes.none

    #     # --- progress ---
    #     if enable_progress:
    #         from srl.runner.callbacks.print_progress import PrintProgress

    #         callbacks.append(
    #             PrintProgress(
    #                 start_time=progress_start_time,
    #                 interval_limit=progress_interval_limit,
    #                 progress_env_info=progress_env_info,
    #                 progress_train_info=progress_train_info,
    #                 progress_worker_info=progress_worker_info,
    #                 progress_worker=progress_worker,
    #                 progress_max_actor=progress_max_actor,
    #                 enable_eval=enable_eval,
    #                 eval_env_sharing=eval_env_sharing,
    #                 eval_episode=eval_episode,
    #                 eval_timeout=eval_timeout,
    #                 eval_max_steps=eval_max_steps,
    #                 eval_players=eval_players,
    #                 eval_shuffle_player=eval_shuffle_player,
    #             )
    #         )
    #         logger.info("add callback PrintProgress")
    #     # ----------------

    #     self._base_run_play_before(
    #         enable_checkpoint_load=True,
    #         enable_checkpoint=True,
    #         enable_history_on_memory=False,
    #         enable_history_on_file=True,
    #         callbacks=callbacks,
    #     )

    #     from .core_mp_debug import train

    #     train(self, choice_method)

    #     self._base_run_play_after()

    def train_distribution(
        self,
        redis_params: "RedisParameters",
        # mp
        actor_num: int = 1,
        queue_capacity: int = 1000,
        trainer_parameter_send_interval: int = 1,
        actor_parameter_sync_interval: int = 1,
        enable_prepare_sample_batch: bool = False,
        enable_trainer_thread: bool = True,
        enable_actor_thread: bool = True,
        # --- stop config
        max_episodes: int = -1,
        timeout: float = -1,
        max_steps: int = -1,
        max_train_count: int = -1,
        max_memory: int = -1,
        # --- play config
        shuffle_player: bool = True,
        # --- progress
        enable_progress: bool = True,
        progress_interval: int = 60 * 1,
        progress_worker: int = 0,
        progress_max_actor: int = 5,
        # --- eval
        enable_eval: bool = True,
        eval_env_sharing: bool = True,
        eval_episode: int = 1,
        eval_timeout: float = -1,
        eval_max_steps: int = -1,
        eval_players: List[Union[None, StrWorkerType, RLWorkerType]] = [],
        eval_shuffle_player: bool = False,
        # --- other
        callbacks: List[Union[CallbackType, "DistributionCallback"]] = [],
    ):
        from .distribution.callback import DistributionCallback

        callbacks_run = cast(List[CallbackType], [c for c in callbacks if issubclass(c.__class__, CallbackType)])
        callbacks_dist = cast(
            List[DistributionCallback], [c for c in callbacks if issubclass(c.__class__, DistributionCallback)]
        )

        self.context.actor_num = actor_num
        self.config.dist_queue_capacity = queue_capacity
        self.config.trainer_parameter_send_interval = trainer_parameter_send_interval
        self.config.actor_parameter_sync_interval = actor_parameter_sync_interval
        self.config.dist_enable_prepare_sample_batch = enable_prepare_sample_batch
        self.config.dist_enable_trainer_thread = enable_trainer_thread
        self.config.dist_enable_actor_thread = enable_actor_thread

        # --- set context
        self.context.run_name = RunNameTypes.main
        # stop config
        self.context.max_episodes = max_episodes
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = max_train_count
        self.context.max_memory = max_memory
        # play config
        self.context.shuffle_player = shuffle_player
        self.context.disable_trainer = False
        # play info
        self.context.distributed = True
        self.context.training = True
        self.context.render_mode = RenderModes.none

        # --- remote progress ---
        if True:
            from srl.runner.callbacks.print_progress import PrintProgress

            callbacks_run.append(
                PrintProgress(
                    progress_worker=progress_worker,
                    progress_max_actor=progress_max_actor,
                    enable_eval=False,
                )
            )
            logger.info("add callback PrintProgress")
        # ----------------

        self._base_run_play_before(
            enable_checkpoint_load=True,
            enable_checkpoint=False,
            enable_history_on_memory=False,
            enable_history_on_file=False,
            callbacks=callbacks_run,
        )

        from srl.runner.distribution.task_manager import TaskManager

        task_manager = TaskManager(redis_params, "client")
        task_manager.create_task(self.create_task_config(), self.make_parameter(is_load=False))
        if self._checkpoint_kwargs is None:
            _k: dict = dict(enable_checkpoint=False)
        else:
            _k: dict = dict(
                enable_checkpoint=False,
                checkpoint_save_dir=self._checkpoint_kwargs["save_dir"],
                checkpoint_interval=self._checkpoint_kwargs["interval"],
            )
        try:
            task_manager.train_wait(
                enable_progress=enable_progress,
                progress_interval=progress_interval,
                **_k,
                enable_eval=enable_eval,
                eval_env_sharing=eval_env_sharing,
                eval_episode=eval_episode,
                eval_timeout=eval_timeout,
                eval_max_steps=eval_max_steps,
                eval_players=eval_players,
                eval_shuffle_player=eval_shuffle_player,
                callbacks=callbacks_dist,
                raise_exception=False,
            )
        finally:
            task_manager.finished("runner")
            task_manager.read_parameter(self.make_parameter(is_load=False))

        self._base_run_play_after()

    def train_distribution_start(
        self,
        redis_params: "RedisParameters",
        # mp
        actor_num: int = 1,
        queue_capacity: int = 1000,
        trainer_parameter_send_interval: int = 1,
        actor_parameter_sync_interval: int = 1,
        enable_prepare_sample_batch: bool = False,
        enable_trainer_thread: bool = True,
        enable_actor_thread: bool = True,
        # --- stop config
        max_episodes: int = -1,
        timeout: float = -1,
        max_steps: int = -1,
        max_train_count: int = -1,
        max_memory: int = -1,
        # --- play config
        shuffle_player: bool = True,
        # --- progress
        progress_worker: int = 0,
        progress_max_actor: int = 5,
        # --- other
        callbacks: List[CallbackType] = [],
    ):
        callbacks = callbacks[:]

        self.context.actor_num = actor_num
        self.config.dist_queue_capacity = queue_capacity
        self.config.trainer_parameter_send_interval = trainer_parameter_send_interval
        self.config.actor_parameter_sync_interval = actor_parameter_sync_interval
        self.config.dist_enable_prepare_sample_batch = enable_prepare_sample_batch
        self.config.dist_enable_trainer_thread = enable_trainer_thread
        self.config.dist_enable_actor_thread = enable_actor_thread

        # --- set context
        self.context.run_name = RunNameTypes.main
        # stop config
        self.context.max_episodes = max_episodes
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = max_train_count
        self.context.max_memory = max_memory
        # play config
        self.context.shuffle_player = shuffle_player
        self.context.disable_trainer = False
        # play info
        self.context.distributed = True
        self.context.training = True
        self.context.render_mode = RenderModes.none

        # --- remote progress ---
        if True:
            from srl.runner.callbacks.print_progress import PrintProgress

            callbacks.append(
                PrintProgress(
                    progress_worker=progress_worker,
                    progress_max_actor=progress_max_actor,
                    enable_eval=False,
                )
            )
            logger.info("add callback PrintProgress")
        # ----------------

        self._base_run_play_before(
            enable_checkpoint_load=True,
            enable_checkpoint=False,
            enable_history_on_memory=False,
            enable_history_on_file=False,
            callbacks=callbacks,
        )

        from srl.runner.distribution.task_manager import TaskManager

        task_manager = TaskManager(redis_params, "client")
        task_manager.create_task(self.create_task_config(), self.make_parameter(is_load=False))

    def evaluate(
        self,
        # --- stop config
        max_episodes: int = 10,
        timeout: float = -1,
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
        callbacks = callbacks[:]

        # --- set context
        self.context.run_name = RunNameTypes.main
        # stop config
        self.context.max_episodes = max_episodes
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.shuffle_player = shuffle_player
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = False
        self.context.render_mode = RenderModes.none

        # --- progress ---
        if enable_progress:
            from srl.runner.callbacks.print_progress import PrintProgress

            callbacks.append(
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
            logger.info("add callback PrintProgress")
        # ----------------

        self._base_run_play_before(
            enable_checkpoint_load=True,
            enable_checkpoint=False,
            enable_history_on_memory=False,
            enable_history_on_file=False,
            callbacks=callbacks,
        )

        state = self.base_run_play(
            trainer_only=False,
            parameter=None,
            memory=None,
            trainer=None,
            workers=None,
        )

        self._base_run_play_after()

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
        timeout: float = -1,
        max_steps: int = -1,
        # --- other
        callbacks: List[CallbackType] = [],
    ):
        callbacks = callbacks[:]
        mode = RenderModes.terminal

        # --- set context
        self.context.run_name = RunNameTypes.main
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = False
        self.context.render_mode = mode

        # --- rendering ---
        from srl.runner.callbacks.rendering import Rendering

        callbacks.append(
            Rendering(
                mode=mode,
                kwargs=render_kwargs,
                step_stop=step_stop,
                render_skip_step=render_skip_step,
            )
        )
        logger.info("enable Rendering")
        # -----------------

        self._base_run_play_before(
            enable_checkpoint_load=True,
            enable_checkpoint=False,
            enable_history_on_memory=False,
            enable_history_on_file=False,
            callbacks=callbacks,
        )
        state = self.base_run_play(
            trainer_only=False,
            parameter=None,
            memory=None,
            trainer=None,
            workers=None,
        )
        self._base_run_play_after()

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
        timeout: float = -1,
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
        callbacks = callbacks[:]
        mode = RenderModes.window

        # --- context
        self.context.run_name = RunNameTypes.main
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = False
        self.context.render_mode = mode

        # --- rendering
        from srl.runner.callbacks.rendering import Rendering

        callbacks.append(
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
        logger.info("add callback Rendering")

        # --- progress ---
        if enable_progress:
            from srl.runner.callbacks.print_progress import PrintProgress

            callbacks.append(
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
            logger.info("add callback PrintProgress")
        # ----------------

        self._base_run_play_before(
            enable_checkpoint_load=True,
            enable_checkpoint=False,
            enable_history_on_memory=False,
            enable_history_on_file=False,
            callbacks=callbacks,
        )
        state = self.base_run_play(
            trainer_only=False,
            parameter=None,
            memory=None,
            trainer=None,
            workers=None,
        )
        self._base_run_play_after()

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
        timeout: float = -1,
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
        callbacks = callbacks[:]
        mode = RenderModes.rgb_array

        # --- set context
        self.context.run_name = RunNameTypes.main
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = False
        self.context.render_mode = mode

        # --- rendering ---
        from srl.runner.callbacks.rendering import Rendering

        rendering = Rendering(
            mode=mode,
            kwargs=render_kwargs,
            step_stop=step_stop,
            render_skip_step=render_skip_step,
            render_interval=render_interval,
            render_scale=1,
            font_name=font_name,
            font_size=font_size,
        )
        callbacks.append(rendering)
        logger.info("add callback Rendering")
        # -----------------

        # --- progress ---
        if enable_progress:
            from srl.runner.callbacks.print_progress import PrintProgress

            callbacks.append(
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
            logger.info("add callback PrintProgress")
        # ----------------

        self._base_run_play_before(
            enable_checkpoint_load=True,
            enable_checkpoint=False,
            enable_history_on_memory=False,
            enable_history_on_file=False,
            callbacks=callbacks,
        )
        state = self.base_run_play(
            trainer_only=False,
            parameter=None,
            memory=None,
            trainer=None,
            workers=None,
        )
        rendering.save_gif(path, render_interval, draw_info)

        self._base_run_play_after()

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
        timeout: float = -1,
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
        callbacks = callbacks[:]
        mode = RenderModes.rgb_array

        # --- set context
        self.context.run_name = RunNameTypes.main
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = False
        self.context.render_mode = mode

        # --- rendering ---
        from srl.runner.callbacks.rendering import Rendering

        rendering = Rendering(
            mode=mode,
            kwargs=render_kwargs,
            step_stop=step_stop,
            render_skip_step=render_skip_step,
            render_interval=render_interval,
            render_scale=render_scale,
            font_name=font_name,
            font_size=font_size,
        )
        callbacks.append(rendering)
        logger.info("add callback Rendering")
        # -----------------

        # --- progress ---
        if enable_progress:
            from srl.runner.callbacks.print_progress import PrintProgress

            callbacks.append(
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
            logger.info("add callback PrintProgress")
        # ----------------

        self._base_run_play_before(
            enable_checkpoint_load=True,
            enable_checkpoint=False,
            enable_history_on_memory=False,
            enable_history_on_file=False,
            callbacks=callbacks,
        )

        state = self.base_run_play(
            trainer_only=False,
            parameter=None,
            memory=None,
            trainer=None,
            workers=None,
        )

        rendering.display(render_interval, render_scale, draw_info)

        self._base_run_play_after()

        return state.episode_rewards_list[0]

    def replay_window(
        self,
        # --- stop config
        timeout: float = -1,
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
        callbacks = callbacks[:]
        mode = RenderModes.rgb_array

        # --- set context
        self.context.run_name = RunNameTypes.main
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = False
        self.context.render_mode = mode

        # --- progress ---
        if enable_progress:
            from srl.runner.callbacks.print_progress import PrintProgress

            callbacks.append(
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
            logger.info("add callback PrintProgress")
        # ----------------

        self._base_run_play_before(
            enable_checkpoint_load=True,
            enable_checkpoint=False,
            enable_history_on_memory=False,
            enable_history_on_file=False,
            callbacks=callbacks,
        )

        from srl.runner.game_windows.replay_window import RePlayableGame

        window = RePlayableGame(self, _is_test)
        window.play()

        self._base_run_play_after()

    def play_terminal(
        self,
        players: List[Union[None, StrWorkerType, RLWorkerType]] = ["human"],
        # Rendering
        render_kwargs: dict = {},
        step_stop: bool = False,
        render_skip_step: bool = True,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- other
        callbacks: List[CallbackType] = [],
    ):
        callbacks = callbacks[:]
        mode = RenderModes.terminal
        self.context.players = players

        # --- set context
        self.context.run_name = RunNameTypes.main
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
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
        callbacks.append(rendering)
        logger.info("add callback Rendering")
        # -----------------

        self._base_run_play_before(
            enable_checkpoint_load=False,
            enable_checkpoint=False,
            enable_history_on_memory=False,
            enable_history_on_file=False,
            callbacks=callbacks,
        )

        state = self.base_run_play(
            trainer_only=False,
            parameter=None,
            memory=None,
            trainer=None,
            workers=None,
            callbacks=callbacks,
        )

        self._base_run_play_after()

        return state.episode_rewards_list[0]

    def play_window(
        self,
        key_bind: Any = None,
        enable_memory: bool = False,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # other
        callbacks: List[CallbackType] = [],
        _is_test: bool = False,  # for test
    ):
        callbacks = callbacks[:]
        mode = RenderModes.rgb_array

        # --- set context
        self.context.run_name = RunNameTypes.main
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = enable_memory
        self.context.render_mode = mode

        self._base_run_play_before(
            enable_checkpoint_load=False,
            enable_checkpoint=False,
            enable_history_on_memory=False,
            enable_history_on_file=False,
            callbacks=callbacks,
        )

        from srl.utils.common import is_packages_installed

        error_text = "This run requires installation of 'PIL', 'pygame'. "
        error_text += "(pip install pillow pygame)"
        assert is_packages_installed(["PIL", "pygame"]), error_text

        from srl.runner.game_windows.playable_game import PlayableGame

        game = PlayableGame(
            self,
            key_bind,
            enable_memory=enable_memory,
            _is_test=_is_test,
        )
        game.play()

        self._base_run_play_after()
