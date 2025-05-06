import logging
import os
from dataclasses import dataclass
from typing import List, Union, cast

from srl.base.define import PlayerType
from srl.base.run.callback import RunCallback
from srl.base.run.core_play import RunStateActor, play
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
        # --- play config
        players: List[PlayerType] = [],
        shuffle_player: bool = True,
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[RunCallback] = [],
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
            callbacks (List[RunCallback], optional): callbacks. Defaults to [].
        """
        callbacks = callbacks[:]

        # --- set context
        self.context.flow_mode = "train"
        # stop config
        self.context.max_episodes = max_episodes
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = max_train_count
        self.context.max_memory = max_memory
        # play config
        self.context.players = players
        self.context.shuffle_player = shuffle_player
        self.context.disable_trainer = False
        # play info
        self.context.distributed = False
        self.context.training = True
        self.context.train_only = False
        self.context.rollout = False
        self.context.env_render_mode = ""
        self.context.rl_render_mode = ""

        if enable_progress:
            self.apply_progress(callbacks, enable_eval=True)
        self.apply_checkpoint(callbacks)
        self._apply_history(callbacks)
        self.apply_mlflow(callbacks)

        self.setup_process()
        play(
            self.context,
            self.state,
            self._parameter_dat,
            self._memory_dat,
            callbacks,
        )
        self._parameter_dat = None
        self._memory_dat = None

        self._after_history()
        return cast(RunStateActor, self.state)

    def rollout(
        self,
        # --- stop config
        max_episodes: int = -1,
        timeout: float = -1,
        max_steps: int = -1,
        max_memory: int = -1,
        # --- play config
        players: List[PlayerType] = [],
        shuffle_player: bool = True,
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[RunCallback] = [],
    ):
        """collect_memory"""
        callbacks = callbacks[:]

        # --- set context
        self.context.flow_mode = "rollout"
        # stop config
        self.context.max_episodes = max_episodes
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = max_memory
        # play config
        self.context.players = players
        self.context.shuffle_player = shuffle_player
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = True
        self.context.train_only = False
        self.context.rollout = True
        self.context.env_render_mode = ""
        self.context.rl_render_mode = ""

        if enable_progress:
            self.apply_progress(callbacks, enable_eval=False)
        self.apply_checkpoint(callbacks)
        self._apply_history(callbacks)
        self.apply_mlflow(callbacks)

        self.setup_process()
        play(
            self.context,
            self.state,
            self._parameter_dat,
            self._memory_dat,
            callbacks,
        )
        self._parameter_dat = None
        self._memory_dat = None

        self._after_history()
        return cast(RunStateActor, self.state)

    def train_only(
        self,
        # --- stop config
        timeout: float = -1,
        max_train_count: int = -1,
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[RunCallback] = [],
    ):
        """Trainerが学習するだけでWorkerによるシミュレーションはありません。"""
        callbacks = callbacks[:]

        # --- context
        self.context.flow_mode = "train_only"
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
        self.context.train_only = True
        self.context.rollout = False
        self.context.env_render_mode = ""
        self.context.rl_render_mode = ""

        if enable_progress:
            self.apply_progress(callbacks, enable_eval=True)
        self.apply_checkpoint(callbacks)
        self._apply_history(callbacks)
        self.apply_mlflow(callbacks)

        from srl.base.run.core_train_only import RunStateTrainer, play_trainer_only

        self.setup_process()
        play_trainer_only(
            self.context,
            self.state,
            self._parameter_dat,
            self._memory_dat,
            callbacks=callbacks,
        )
        self._parameter_dat = None
        self._memory_dat = None

        self._after_history()
        return cast(RunStateTrainer, self.state)

    def train_mp(
        self,
        # mp
        actor_num: int = 1,
        queue_capacity: int = 1000,
        trainer_parameter_send_interval: float = 1,
        actor_parameter_sync_interval: float = 1,
        actor_devices: Union[str, List[str]] = "AUTO",
        enable_mp_memory: bool = True,
        train_to_mem_queue_capacity: int = 100,
        mem_to_train_queue_capacity: int = 10,
        # --- stop config
        timeout: float = -1,
        max_train_count: int = -1,
        # --- play config
        players: List[PlayerType] = [],
        shuffle_player: bool = True,
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[RunCallback] = [],
    ):
        """multiprocessingを使用した分散学習による学習を実施します。"""
        callbacks = callbacks[:]

        # --- mp config
        self.context.actor_num = actor_num
        self.context.actor_devices = actor_devices

        # --- set context
        self.context.flow_mode = "train_mp"
        # stop config
        self.context.max_episodes = -1
        self.context.timeout = timeout
        self.context.max_steps = -1
        self.context.max_train_count = max_train_count
        self.context.max_memory = -1
        # play config
        self.context.players = players
        self.context.shuffle_player = shuffle_player
        self.context.disable_trainer = False
        # play info
        self.context.distributed = True
        self.context.training = True
        self.context.train_only = False
        self.context.rollout = False
        self.context.env_render_mode = ""
        self.context.rl_render_mode = ""

        if enable_progress:
            self.apply_progress(callbacks, enable_eval=True)
        self.apply_checkpoint(callbacks)
        self._apply_history(callbacks)
        self.apply_mlflow(callbacks)

        # ---
        if self.rl_config.get_framework() == "tensorflow":
            os.environ["SRL_TF_GPU_INITIALIZE_DEVICES"] = "1"
        params_dat = self._parameter_dat
        if (params_dat is None) and (self.state.parameter is not None):
            params_dat = self.state.parameter.backup(to_cpu=True)
        memory_dat = self._memory_dat
        if (memory_dat is None) and (self.state.memory is not None):
            memory_dat = self.state.memory.backup(to_cpu=True)

        if enable_mp_memory:
            from srl.base.run.play_mp_memory import MpConfig, train

            params_dat, memory_dat = train(
                MpConfig(
                    self.context,
                    callbacks,
                    queue_capacity=queue_capacity,
                    trainer_parameter_send_interval=trainer_parameter_send_interval,
                    actor_parameter_sync_interval=actor_parameter_sync_interval,
                    train_to_mem_queue_capacity=train_to_mem_queue_capacity,
                    mem_to_train_queue_capacity=mem_to_train_queue_capacity,
                ),
                params_dat,
                memory_dat,
            )
            self._parameter_dat = params_dat
            self._memory_dat = memory_dat

        else:
            from srl.base.run.play_mp import MpConfig, train

            params_dat, memory_dat = train(
                MpConfig(
                    self.context,
                    callbacks,
                    queue_capacity=queue_capacity,
                    trainer_parameter_send_interval=trainer_parameter_send_interval,
                    actor_parameter_sync_interval=actor_parameter_sync_interval,
                ),
                params_dat,
                memory_dat,
            )
            self._parameter_dat = params_dat
            self._memory_dat = memory_dat

        self._after_history()

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
