import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, List, Optional, Union, cast

import numpy as np

from srl.base.context import RunContext
from srl.base.define import EnvObservationType, PlayersType, RLObservationType
from srl.base.rl.config import TRLConfig
from srl.base.rl.parameter import RLParameter
from srl.base.run.callback import RunCallback
from srl.base.run.core_play import RunStateActor, play
from srl.runner.runner_base import RunnerBase

if TYPE_CHECKING:
    from srl.runner.distribution.callback import DistributionCallback
    from srl.runner.distribution.connectors.redis_ import RedisParameters

logger = logging.getLogger(__name__)


@dataclass
class Runner(Generic[TRLConfig], RunnerBase[TRLConfig]):
    @staticmethod
    def create(context: RunContext):
        context = context.copy()
        return Runner(context.env_config, context.rl_config, context)

    def core_play(
        self,
        enable_progress: bool = False,
        callbacks: List[RunCallback] = [],
    ):
        """設定されているcontextでそのままplayする"""
        callbacks = callbacks[:]
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
        return self.state

    # --------------------------------------------
    # train
    # --------------------------------------------
    def train(
        self,
        # --- stop config
        max_episodes: int = -1,
        timeout: float = -1,
        max_steps: int = -1,
        max_train_count: int = -1,
        max_memory: int = -1,
        # --- play config
        players: PlayersType = [],
        shuffle_player: bool = True,
        # --- train option
        train_interval: int = 1,
        train_repeat: int = 1,
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
            train_interval (int, optional): 学習間隔（step）. Defaults to 1.
            train_repeat (int, optional): 1stepあたりの学習回数. Defaults to 1.
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
        # train option
        self.context.train_interval = train_interval
        self.context.train_repeat = train_repeat
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
        players: PlayersType = [],
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
        mem_to_train_queue_capacity: int = 5,
        # memory
        return_memory_data: bool = False,
        return_memory_timeout: int = 60 * 60 * 1,
        # --- stop config
        timeout: float = -1,
        max_train_count: int = -1,
        # --- play config
        players: PlayersType = [],
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

        if not self.rl_config.is_setup():
            self.rl_config.setup(self.make_env())

        # ---
        if self.rl_config.get_framework() == "tensorflow":
            os.environ["SRL_TF_GPU_INITIALIZE_DEVICES"] = "1"
        params_dat = self._parameter_dat
        if (params_dat is None) and (self.state.parameter is not None):
            params_dat = self.state.parameter.backup(serialized=True)
        memory_dat = self._memory_dat
        if (memory_dat is None) and (self.state.memory is not None):
            memory_dat = self.state.memory.backup(compress=True)

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
                    return_memory_data=return_memory_data,
                    return_memory_timeout=return_memory_timeout,
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
                    return_memory_data=return_memory_data,
                    return_memory_timeout=return_memory_timeout,
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

    # --------------------------------------------
    # distribution
    # --------------------------------------------
    def train_distribution(
        self,
        redis_params: "RedisParameters",
        # mp
        actor_num: int = 1,
        queue_capacity: int = 1000,
        trainer_parameter_send_interval: int = 1,
        actor_parameter_sync_interval: int = 1,
        enable_trainer_thread: bool = True,
        enable_actor_thread: bool = True,
        # --- stop config
        timeout: float = -1,
        max_train_count: int = -1,
        # --- play config
        players: PlayersType = [],
        shuffle_player: bool = True,
        # --- progress
        enable_progress: bool = True,
        progress_interval: int = 60 * 1,
        # --- other
        callbacks: List[Union[RunCallback, "DistributionCallback"]] = [],
    ):
        from srl.runner.distribution.callback import DistributionCallback

        callbacks_dist: List[DistributionCallback] = []
        callbacks_run: List[RunCallback] = []
        for c in callbacks:
            if issubclass(c.__class__, DistributionCallback):
                callbacks_dist.append(cast(DistributionCallback, c))
            else:
                callbacks_run.append(cast(RunCallback, c))

        # --- mp config
        self.context.actor_num = actor_num

        # --- set context
        self.context.flow_mode = "train_distribution"
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
            self.apply_progress(callbacks_run, enable_eval=False)

        if not self.rl_config.is_setup():
            self.rl_config.setup(self.make_env())

        from srl.runner.distribution.task_manager import TaskConfig, TaskManager

        task_manager = TaskManager(redis_params, "client")
        task_manager.create_task(
            TaskConfig(
                self.context,
                callbacks_run,
                queue_capacity=queue_capacity,
                trainer_parameter_send_interval=trainer_parameter_send_interval,
                actor_parameter_sync_interval=actor_parameter_sync_interval,
                enable_trainer_thread=enable_trainer_thread,
                enable_actor_thread=enable_actor_thread,
            ),
            self.make_parameter(),
        )

        try:
            task_manager.train_wait(
                enable_progress=enable_progress,
                progress_interval=progress_interval,
                progress_kwargs=self._progress_kwargs,
                checkpoint_kwargs=self._checkpoint_kwargs,
                history_on_file_kwargs=self._history_on_file_kwargs,
                callbacks=callbacks_dist,
                raise_exception=False,
            )
        finally:
            task_manager.finished("runner")
            task_manager.read_parameter(self.make_parameter())

    def train_distribution_start(
        self,
        redis_params: "RedisParameters",
        # mp
        actor_num: int = 1,
        queue_capacity: int = 1000,
        trainer_parameter_send_interval: int = 1,
        actor_parameter_sync_interval: int = 1,
        enable_trainer_thread: bool = True,
        enable_actor_thread: bool = True,
        # --- stop config
        timeout: float = -1,
        max_train_count: int = -1,
        # --- play config
        players: PlayersType = [],
        shuffle_player: bool = True,
        # --- other
        enable_progress: bool = True,
        callbacks: List[RunCallback] = [],
    ):
        callbacks = callbacks[:]

        # --- mp config
        self.context.actor_num = actor_num

        # --- set context
        self.context.flow_mode = "train_distribution_start"
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
        self.context.env_render_mode = ""
        self.context.rl_render_mode = ""

        if enable_progress:
            self.apply_progress(callbacks, enable_eval=False)

        if not self.rl_config.is_setup():
            self.rl_config.setup(self.make_env())

        from srl.runner.distribution.task_manager import TaskConfig, TaskManager

        task_manager = TaskManager(redis_params, "client")
        task_manager.create_task(
            TaskConfig(
                self.context,
                callbacks,
                queue_capacity=queue_capacity,
                trainer_parameter_send_interval=trainer_parameter_send_interval,
                actor_parameter_sync_interval=actor_parameter_sync_interval,
                enable_trainer_thread=enable_trainer_thread,
                enable_actor_thread=enable_actor_thread,
            ),
            self.make_parameter(),
        )

    # --------------------------------------------
    # play
    # --------------------------------------------
    def evaluate(
        self,
        # --- stop config
        max_episodes: int = 10,
        timeout: float = -1,
        max_steps: int = -1,
        # --- play config
        players: PlayersType = [],
        shuffle_player: bool = True,
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[RunCallback] = [],
    ) -> Union[List[float], List[List[float]]]:  # single play , multi play
        """シミュレーションし、報酬を返します。

        Args:
            max_episodes (int, optional): 終了するまでのエピソード数. Defaults to 10.
            timeout (int, optional): 終了するまでの時間（秒）. Defaults to -1.
            max_steps (int, optional): 終了するまでの総ステップ. Defaults to -1.
            shuffle_player (bool, optional): playersをシャッフルするかどうか. Defaults to True.
            enable_progress (bool, optional): 進捗を表示するか. Defaults to True.
            callbacks (List[RunCallback], optional): callbacks. Defaults to [].

        Returns:
            Union[List[float], List[List[float]]]: プレイヤー数が1人なら Lost[float]、複数なら List[List[float]]] を返します。
        """
        callbacks = callbacks[:]

        # --- set context
        self.context.flow_mode = "evaluate"
        # stop config
        self.context.max_episodes = max_episodes
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.players = players
        self.context.shuffle_player = shuffle_player
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = False
        self.context.train_only = False
        self.context.rollout = False
        self.context.env_render_mode = ""
        self.context.rl_render_mode = ""

        if enable_progress:
            self.apply_progress(callbacks, enable_eval=False)

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

        state = cast(RunStateActor, self.state)
        if self.env.player_num == 1:
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
        # --- play config
        players: PlayersType = [],
        # --- other
        training_flag: bool = False,
        callbacks: List[RunCallback] = [],
    ):
        callbacks = callbacks[:]

        # --- set context
        self.context.flow_mode = "render_terminal"
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.players = players
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = training_flag
        self.context.train_only = False
        self.context.rollout = False
        # render_modeはRendering側で設定
        # self.context.env_render_mode = ""
        # self.context.rl_render_mode = ""

        if training_flag:
            params_dat = self.parameter.backup()

        # --- rendering ---
        from srl.runner.callbacks.rendering import Rendering

        callbacks.append(
            Rendering(
                mode="terminal",
                kwargs=render_kwargs,
                step_stop=step_stop,
                render_skip_step=render_skip_step,
            )
        )
        # -----------------

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

        if training_flag:
            self.parameter.restore(params_dat)

        state = cast(RunStateActor, self.state)
        return state.episode_rewards_list[0]

    def render_window(
        self,
        # rendering
        render_kwargs: dict = {},
        render_skip_step: bool = True,
        # render option
        render_interval: float = -1,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- play config
        players: PlayersType = [],
        # --- progress
        enable_progress: bool = True,
        # --- other
        training_flag: bool = False,
        callbacks: List[RunCallback] = [],
    ):
        callbacks = callbacks[:]

        # --- context
        self.context.flow_mode = "render_window"
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.players = players
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = training_flag
        self.context.train_only = False
        self.context.rollout = False
        # render_modeはRendering側で設定
        # self.context.env_render_mode = ""
        # self.context.rl_render_mode = ""

        if training_flag:
            params_dat = self.parameter.backup()

        # --- rendering
        from srl.runner.callbacks.rendering import Rendering

        rendering = Rendering(
            mode="window",
            kwargs=render_kwargs,
            step_stop=False,
            render_interval=render_interval,
            render_skip_step=render_skip_step,
        )
        callbacks.append(rendering)

        if enable_progress:
            self.apply_progress(callbacks, enable_eval=False)

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

        if training_flag:
            self.parameter.restore(params_dat)

    def run_render(
        self,
        # rendering
        render_kwargs: dict = {},
        step_stop: bool = False,
        render_skip_step: bool = True,
        # render option
        render_worker: int = 0,
        render_add_rl_terminal: bool = True,
        render_add_rl_rgb: bool = True,
        render_add_info_text: bool = True,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- play config
        players: PlayersType = [],
        # --- progress
        enable_progress: bool = True,
        # --- other
        training_flag: bool = False,
        callbacks: List[RunCallback] = [],
    ):
        callbacks = callbacks[:]

        # --- set context
        self.context.flow_mode = "run_render"
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.players = players
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = training_flag
        self.context.train_only = False
        self.context.rollout = False
        # render_modeはRendering側で設定
        # self.context.env_render_mode = ""
        # self.context.rl_render_mode = ""

        if training_flag:
            params_dat = self.parameter.backup()

        # --- rendering ---
        from srl.runner.callbacks.rendering import Rendering

        rendering = Rendering(
            mode="rgb_array",
            kwargs=render_kwargs,
            step_stop=step_stop,
            render_skip_step=render_skip_step,
            render_worker=render_worker,
            render_add_rl_terminal=render_add_rl_terminal,
            render_add_rl_rgb=render_add_rl_rgb,
            render_add_info_text=render_add_info_text,
        )
        callbacks.append(rendering)
        # -----------------

        if enable_progress:
            self.apply_progress(callbacks, enable_eval=False)

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

        if training_flag:
            self.parameter.restore(params_dat)

        if self.context.run_name != "eval":
            logger.info(f"render step: {self.state.total_step}, reward: {self.state.episode_rewards_list}")
        return rendering

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
        render_worker: int = 0,
        render_add_rl_terminal: bool = True,
        render_add_rl_rgb: bool = True,
        render_add_info_text: bool = True,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- play config
        players: PlayersType = [],
        # --- progress
        enable_progress: bool = True,
        # --- other
        training_flag: bool = False,
        callbacks: List[RunCallback] = [],
    ):
        render = self.run_render(
            render_kwargs,
            step_stop,
            render_skip_step,
            render_worker,
            render_add_rl_terminal,
            render_add_rl_rgb,
            render_add_info_text,
            timeout,
            max_steps,
            players,
            enable_progress,
            training_flag,
            callbacks,
        )
        render.save_gif(path, render_interval, render_scale)
        return render

    def animation_save_avi(
        self,
        path: str,
        # rendering
        render_kwargs: dict = {},
        step_stop: bool = False,
        render_skip_step: bool = True,
        # render option
        render_interval: float = -1,  # ms
        render_scale: float = 1.0,
        render_worker: int = 0,
        render_add_rl_terminal: bool = True,
        render_add_rl_rgb: bool = True,
        render_add_info_text: bool = True,
        codec: str = "XVID",
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- play config
        players: PlayersType = [],
        # --- progress
        enable_progress: bool = True,
        # --- other
        training_flag: bool = False,
        callbacks: List[RunCallback] = [],
    ):
        render = self.run_render(
            render_kwargs,
            step_stop,
            render_skip_step,
            render_worker,
            render_add_rl_terminal,
            render_add_rl_rgb,
            render_add_info_text,
            timeout,
            max_steps,
            players,
            enable_progress,
            training_flag,
            callbacks,
        )
        render.save_avi(path, render_interval, render_scale, codec=codec)
        return render

    def animation_display(
        self,
        # rendering
        render_kwargs: dict = {},
        step_stop: bool = False,
        render_skip_step: bool = True,
        # render option
        render_interval: float = -1,  # ms
        render_scale: float = 1.0,
        render_worker: int = 0,
        render_add_rl_terminal: bool = True,
        render_add_rl_rgb: bool = True,
        render_add_info_text: bool = True,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- play config
        players: PlayersType = [],
        # --- progress
        enable_progress: bool = True,
        # --- other
        training_flag: bool = False,
        callbacks: List[RunCallback] = [],
    ):
        render = self.run_render(
            render_kwargs,
            step_stop,
            render_skip_step,
            render_worker,
            render_add_rl_terminal,
            render_add_rl_rgb,
            render_add_info_text,
            timeout,
            max_steps,
            players,
            enable_progress,
            training_flag,
            callbacks,
        )
        render.display(render_interval, render_scale)
        return render

    def replay_window(
        self,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- play config
        players: PlayersType = [],
        # --- progress
        enable_progress: bool = True,
        # --- other
        training_flag: bool = False,
        render_player: int = 0,
        print_state: bool = True,
        callbacks: List[RunCallback] = [],
        _is_test: bool = False,  # for test
    ):
        callbacks = callbacks[:]

        # --- set context
        self.context.flow_mode = "replay_window"
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.players = players
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = training_flag
        self.context.train_only = False
        self.context.rollout = False
        # render_modeはRePlayableGame側で設定
        # self.context.env_render_mode = ""
        # self.context.rl_render_mode = ""

        if training_flag:
            params_dat = self.parameter.backup()

        if enable_progress:
            self.apply_progress(callbacks, enable_eval=False)

        from srl.runner.game_windows.replay_window import RePlayableGame

        self.setup_process()
        self.state.parameter = self.make_parameter()
        window = RePlayableGame(
            self.context,
            self.state,
            render_player=render_player,
            print_state=print_state,
            callbacks=callbacks,
            _is_test=_is_test,
        )
        window.play()

        if training_flag:
            self.parameter.restore(params_dat)

    def play_terminal(
        self,
        action_division_num: int = 5,
        enable_memory: bool = False,
        players: PlayersType = [],
        # Rendering
        render_kwargs: dict = {},
        step_stop: bool = False,
        render_skip_step: bool = True,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- other
        training_flag: bool = False,
        callbacks: List[RunCallback] = [],
    ):
        callbacks = callbacks[:]

        # --- set context
        self.context.flow_mode = "play_terminal"
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.players = players
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = enable_memory
        self.context.train_only = False
        self.context.rollout = enable_memory
        # render_modeはRendering側で設定
        # self.context.env_render_mode = ""
        # self.context.rl_render_mode = ""

        if training_flag:
            params_dat = self.parameter.backup()

        # --- rendering ---
        from srl.runner.callbacks.rendering import Rendering

        rendering = Rendering(
            mode="terminal",
            kwargs=render_kwargs,
            step_stop=step_stop,
            render_skip_step=render_skip_step,
            render_env=False,
        )
        callbacks.append(rendering)

        # -----------------
        from srl.runner.callbacks.manual_play_callback import ManualPlayCallback

        callbacks.append(ManualPlayCallback(self.make_env(), action_division_num))
        # -----------------

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

        if training_flag:
            self.parameter.restore(params_dat)

        return self.state.episode_rewards_list[0]

    def play_window(
        self,
        key_bind: Any = None,
        view_state: bool = True,
        action_division_num: int = 5,
        enable_memory: bool = False,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- play config
        players: PlayersType = [],
        # --- other
        training_flag: bool = False,
        callbacks: List[RunCallback] = [],
        _is_test: bool = False,  # for test
    ):
        callbacks2 = cast(List[RunCallback], [c for c in callbacks if issubclass(c.__class__, RunCallback)])

        # --- set context
        self.context.flow_mode = "play_window"
        # stop config
        self.context.max_episodes = -1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.players = players
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = enable_memory or training_flag
        self.context.train_only = False
        self.context.rollout = enable_memory
        # render_modeはPlayableGame側で設定
        # self.context.env_render_mode = ""
        # self.context.rl_render_mode = ""

        if training_flag:
            params_dat = self.parameter.backup()

        from srl.utils.common import is_packages_installed

        error_text = "This run requires installation of 'PIL', 'pygame'. "
        error_text += "(pip install pillow pygame)"
        assert is_packages_installed(["PIL", "pygame"]), error_text

        from srl.runner.game_windows.playable_game import PlayableGame

        self.setup_process()
        game = PlayableGame(
            env=self.make_env(),
            context=self.context,
            worker=self.make_worker(),
            view_state=view_state,
            action_division_num=action_division_num,
            key_bind=key_bind,
            callbacks=callbacks2,
            _is_test=_is_test,
        )
        game.play()

        if training_flag:
            self.parameter.restore(params_dat)

    # --------------------------------------------
    # utils
    # --------------------------------------------
    def print_config(self):
        import pprint

        print(f"env\n{pprint.pformat(self.env_config.to_dict())}")
        print(f"rl\n{pprint.pformat(self.rl_config.to_dict())}")
        print(f"context\n{pprint.pformat(self.context.to_dict())}")

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

    def summary(self, summary_parameter: bool = False):
        if self.rl_config.is_setup():
            self.rl_config.setup(self.make_env())
        self.env_config.summary()
        self.rl_config.summary()
        if summary_parameter:
            parameter = self.make_parameter()
            parameter.summary()

    def get_env_init_state(self, encode: bool = True) -> Union[EnvObservationType, RLObservationType]:
        env = self.make_env()
        env.setup()
        env.reset()
        state = env.state
        if encode:
            worker = self.make_worker()
            state = worker._config.state_encode_one_step(state, env)
        return state

    def evaluate_compare_to_baseline_single_player(
        self,
        episode: int = -1,
        baseline: Optional[float] = None,
        eval_kwargs: dict = {},
        enable_backup_restore: bool = True,
    ):
        # baseline
        env = self.make_env()
        assert env.player_num == 1

        if episode <= 0:
            assert isinstance(env.reward_baseline, dict)
            episode = env.reward_baseline.get("episode", 0)
        if episode <= 0:
            episode = 100

        if baseline is None:
            assert isinstance(env.reward_baseline, dict)
            baseline = env.reward_baseline.get("baseline", None)
        assert baseline is not None, "Please specify a 'baseline'."

        # check restore
        if enable_backup_restore:
            parameter = self.make_parameter()
            parameter.restore(parameter.backup())

        # eval
        rewards = self.evaluate(max_episodes=episode, **eval_kwargs)

        # check
        reward = np.mean(rewards)
        result = reward >= baseline
        logger.info(f"{result}: {reward} >= {baseline}(baseline)")

        return result

    def evaluate_compare_to_baseline_multiplayer(
        self,
        players: PlayersType = [],
        baseline_params: List[dict] = [],
        eval_kwargs: dict = {},
        enable_backup_restore: bool = True,
    ):
        # baseline
        env = self.make_env()
        assert env.player_num > 1

        if baseline_params == []:
            if env.reward_baseline is not None:
                baseline_params = env.reward_baseline
        assert isinstance(baseline_params, list)

        # check restore
        if enable_backup_restore:
            parameter = self.make_parameter()
            parameter.restore(parameter.backup())

        results = []
        for params in baseline_params:
            episode = params.get("episode", 100)
            players = params.get("players", [])
            baseline = params["baseline"]

            # eval
            rewards = self.evaluate(
                max_episodes=episode,
                players=players,
                **eval_kwargs,
            )

            # check
            rewards = np.mean(rewards, axis=0)
            result = []
            logger.info(f"baseline {baseline}, rewards {rewards}")
            for i, reward in enumerate(rewards):
                if baseline[i] is None:
                    result.append(True)
                else:
                    result.append(bool(reward >= baseline[i]))
            logger.info(f"{result=}")
            results.append(result)

        return results
