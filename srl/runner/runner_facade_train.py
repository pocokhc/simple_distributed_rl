import logging
from dataclasses import dataclass
from typing import List, Optional, Union, cast

from srl.base.context import RunNameTypes
from srl.base.define import RenderModes
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.rl.trainer import RLTrainer
from srl.base.run.callback import CallbackType
from srl.base.run.core_play import RunStateActor
from srl.runner.runner_base import RunnerBase

logger = logging.getLogger(__name__)


@dataclass()
class RunnerFacadeTrain(RunnerBase):
    def train(
        self,
        # --- stop config
        max_episodes: int = -1,
        timeout: float = -1,
        max_steps: int = -1,
        max_train_count: int = -1,
        max_memory: int = -1,
        # --- thread
        enable_train_thread: bool = False,
        thread_queue_capacity: int = 10,
        # --- play config
        shuffle_player: bool = True,
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[CallbackType] = [],
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
        trainer: Optional[RLTrainer] = None,
        logger_config: bool = True,
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
            enable_eval (bool, optional): 評価用のシミュレーションを実行します. Defaults to False.
            callbacks (List[CallbackType], optional): callbacks. Defaults to [].
        """
        callbacks = callbacks[:]

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
        self.context.distributed = False
        self.context.training = True
        self.context.train_only = False
        self.context.rendering = False
        self.context.render_mode = RenderModes.none
        # thread
        self.context.enable_train_thread = enable_train_thread
        self.context.thread_queue_capacity = thread_queue_capacity

        self._base_run_before(
            enable_progress=enable_progress,
            enable_progress_eval=True,
            enable_checkpoint=True,
            enable_history_on_memory=True,
            enable_history_on_file=True,
            callbacks=callbacks,
        )
        state = cast(
            RunStateActor,
            self._wrap_base_run(
                parameter=parameter,
                memory=memory,
                trainer=trainer,
                workers=None,
                main_worker_idx=0,
                callbacks=callbacks,
                logger_config=logger_config,
            ),
        )
        self._base_run_after()

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
        # --- other
        callbacks: List[CallbackType] = [],
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
        logger_config: bool = True,
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
        self.context.rendering = False
        self.context.render_mode = RenderModes.none
        # thread
        self.context.enable_train_thread = False

        self._base_run_before(
            enable_progress=enable_progress,
            enable_progress_eval=False,
            enable_checkpoint=True,
            enable_history_on_memory=True,
            enable_history_on_file=True,
            callbacks=callbacks,
        )
        state = cast(
            RunStateActor,
            self._wrap_base_run(
                parameter=parameter,
                memory=memory,
                trainer=None,
                workers=None,
                main_worker_idx=0,
                callbacks=callbacks,
                logger_config=logger_config,
            ),
        )
        self._base_run_after()

        return state

    def train_only(
        self,
        # --- stop config
        timeout: float = -1,
        max_train_count: int = -1,
        # --- thread
        enable_train_thread: bool = False,
        thread_queue_capacity: int = 10,
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[CallbackType] = [],
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
        trainer: Optional[RLTrainer] = None,
        logger_config: bool = True,
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
        self.context.rendering = False
        self.context.render_mode = RenderModes.none
        # thread
        self.context.enable_train_thread = enable_train_thread
        self.context.thread_queue_capacity = thread_queue_capacity

        self._base_run_before(
            enable_progress=enable_progress,
            enable_progress_eval=True,
            enable_checkpoint=True,
            enable_history_on_memory=True,
            enable_history_on_file=True,
            callbacks=callbacks,
        )
        state = self._wrap_base_run_trainer_only(
            parameter=parameter,
            memory=memory,
            trainer=trainer,
            callbacks=callbacks,
            logger_config=logger_config,
        )
        self._base_run_after()

        return state

    def train_mp(
        self,
        # mp
        actor_num: int = 1,
        queue_capacity: int = 1000,
        trainer_parameter_send_interval: int = 1,
        actor_parameter_sync_interval: int = 1,
        actor_devices: Union[str, List[str]] = "AUTO",
        # --- stop config
        timeout: float = -1,
        max_train_count: int = -1,
        # --- thread
        enable_train_thread: bool = False,
        thread_queue_capacity: int = 10,
        # --- play config
        shuffle_player: bool = True,
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[CallbackType] = [],
        logger_config: bool = True,
    ):
        """multiprocessingを使用した分散学習による学習を実施します。"""
        callbacks = callbacks[:]

        # --- mp config
        self.context.actor_num = actor_num
        self.context.actor_devices = actor_devices

        # --- set context
        self.context.run_name = RunNameTypes.main
        # stop config
        self.context.max_episodes = -1
        self.context.timeout = timeout
        self.context.max_steps = -1
        self.context.max_train_count = max_train_count
        self.context.max_memory = -1
        # play config
        self.context.shuffle_player = shuffle_player
        self.context.disable_trainer = False
        # play info
        self.context.distributed = True
        self.context.training = True
        self.context.rendering = False
        self.context.render_mode = RenderModes.none
        # thread
        self.context.enable_train_thread = enable_train_thread
        self.context.thread_queue_capacity = thread_queue_capacity

        self._base_run_before(
            enable_progress=enable_progress,
            enable_progress_eval=True,
            enable_checkpoint=True,
            enable_history_on_memory=False,
            enable_history_on_file=True,
            callbacks=callbacks,
        )

        from srl.base.run.play_mp import MpData, train

        train(
            MpData(
                self.context,
                callbacks,
                queue_capacity=queue_capacity,
                trainer_parameter_send_interval=trainer_parameter_send_interval,
                actor_parameter_sync_interval=actor_parameter_sync_interval,
            ),
            self.make_parameter(),
            self.make_memory(),
            logger_config,
        )

        self._base_run_after()

    # def train_mp_debug(
    #     self,
    #     # mp
    #     actor_num: int = 1,
    #     queue_capacity: int = 1000,
    #     trainer_parameter_send_interval: int = 1,
    #     actor_parameter_sync_interval: int = 1,
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
    #     eval_episode: int = 1,
    #     eval_timeout: float = -1,
    #     eval_max_steps: int = -1,
    #     eval_players: List[PlayerType] = [],
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
    #    self.context.rendering = False
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
    #                 eval_episode=eval_episode,
    #                 eval_timeout=eval_timeout,
    #                 eval_max_steps=eval_max_steps,
    #                 eval_players=eval_players,
    #                 eval_shuffle_player=eval_shuffle_player,
    #             )
    #         )
    #         logger.info("add callback PrintProgress")
    #     # ----------------

    #     self._base_run_before(
    #
    #         enable_checkpoint=True,
    #         enable_history_on_memory=False,
    #         enable_history_on_file=True,
    #         callbacks=callbacks,
    #     )

    #     from .core_mp_debug import train

    #     train(self, choice_method)

    #     self._base_run_play_after()
