import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, List, Optional, Union, cast

import numpy as np

from srl.base.context import RunContext
from srl.base.define import EnvObservationType, PlayersType, RLObservationType
from srl.base.exception import NotSupportedError
from srl.base.rl.config import TRLConfig
from srl.base.rl.parameter import RLParameter
from srl.base.run.callback import RunCallback
from srl.base.run.core_play import RunStateActor, play
from srl.runner.runner_base import RunnerBase

if TYPE_CHECKING:
    from srl.runner.distribution.callback import DistributionCallback
    from srl.runner.distribution.connector_configs import RedisParameters

logger = logging.getLogger(__name__)


@dataclass
class Runner(Generic[TRLConfig], RunnerBase[TRLConfig]):
    def play(self, enable_progress: bool = True):
        """設定されているcontextでそのままplayする

        setup_context: 同じ設定で複数回呼ばれる場合はFalseにする（eval等）
        """
        context = self.context.copy()

        # playが特殊なもの
        if self.context.play_mode == "train_mp":
            self.update_context_train_mp(context, enable_progress)
            self.train_mp(used_context=context)
            return
        elif self.context.play_mode == "train_distribution":
            raise NotSupportedError(context.play_mode)
            self.update_context_train_distribution(context, enable_progress)
            self.train_distribution(used_context=context)
            return
        elif self.context.play_mode == "train_distribution_start":
            raise NotSupportedError(context.play_mode)
            self.update_context_train_distribution(context, enable_progress)
            self.train_distribution_start(used_context=context)
            return
        elif self.context.play_mode == "replay_window":
            raise NotSupportedError(context.play_mode)
        elif self.context.play_mode == "play_window":
            raise NotSupportedError(context.play_mode)

        if self.context.play_mode == "train":
            self.update_context_train(context, enable_progress)
        elif self.context.play_mode == "rollout":
            self.update_context_rollout(context, enable_progress)
        elif self.context.play_mode == "train_only":
            self.update_context_train_only(context, enable_progress)
        elif self.context.play_mode == "evaluate":
            self.update_context_evaluate(context, enable_progress)
        elif self.context.play_mode == "render_terminal":
            raise NotSupportedError(context.play_mode)
        elif self.context.play_mode == "render_window":
            raise NotSupportedError(context.play_mode)
        elif self.context.play_mode == "play_terminal":
            raise NotSupportedError(context.play_mode)
        else:
            raise NotSupportedError(context.play_mode)

        self.state = play(
            context,
            env=self.make_env(),
            worker=self.make_worker(),
            trainer=self.make_trainer() if self.context.training else None,
            workers=None,
        )
        self._after_history()
        return self.state

    def play_direct(self):
        """設定されているcontextでそのままplayする、チェックなし"""
        self.state = play(
            self.context,
            env=self.make_env(),
            worker=self.make_worker(),
            trainer=self.make_trainer() if self.context.training else None,
            workers=None,
        )
        self._after_history()
        return self.state

    # --------------------------------------------
    # train
    # --------------------------------------------
    def train(
        self,
        # --- stop config
        max_episodes: int = 0,
        timeout: float = 0,
        max_steps: int = 0,
        max_train_count: int = 0,
        max_memory: int = 0,
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
            players (PlayerTypes, optional): 二人以上の環境で他プレイヤーのアルゴリズム
            shuffle_player (bool, optional): playersをシャッフルするかどうか. Defaults to True.
            train_interval (int, optional): 学習間隔（step）. Defaults to 1.
            train_repeat (int, optional): 1stepあたりの学習回数. Defaults to 1.
            enable_progress (bool, optional): 進捗を表示するか. Defaults to True.
            callbacks (List[RunCallback], optional): callbacks. Defaults to [].
        """
        c = self.context.copy()
        c.max_episodes = max_episodes
        c.timeout = timeout
        c.max_steps = max_steps
        c.max_train_count = max_train_count
        c.max_memory = max_memory
        c.players = players
        c.shuffle_player = shuffle_player
        c.train_interval = train_interval
        c.train_repeat = train_repeat
        self.update_context_train(c, enable_progress)
        c.callbacks += callbacks[:]

        self.state = play(
            c,
            env=self.make_env(),
            worker=self.make_worker(),
            trainer=self.make_trainer(),
            workers=None,
        )

        self._after_history()
        return cast(RunStateActor, self.state)

    def update_context_train(self, context: Optional[RunContext] = None, enable_progress: bool = False):
        c = self.context if context is None else context
        c.play_mode = "train"
        # --- stop config
        # c.max_episodes = max_episodes
        # c.timeout = timeout
        # c.max_steps = max_steps
        # c.max_train_count = max_train_count
        # c.max_memory = max_memory
        # --- play config
        # c.players = players
        # c.shuffle_player = shuffle_player
        c.disable_trainer = False
        # --- train option
        # c.train_interval = train_interval
        # c.train_repeat = train_repeat
        # --- play info
        c.distributed = False
        c.training = True
        c.train_only = False
        c.rollout = False
        # --- render
        c.env_render_mode = ""
        c.rl_render_mode = ""

        if enable_progress:
            self.apply_progress(c.callbacks, apply_eval=True)
        self.apply_checkpoint(c.callbacks)
        self._apply_history(c.callbacks)
        self.apply_mlflow(c.callbacks)
        return c

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
        c = self.context.copy()
        c.max_episodes = max_episodes
        c.timeout = timeout
        c.max_steps = max_steps
        c.max_memory = max_memory
        c.players = players
        c.shuffle_player = shuffle_player
        self.update_context_rollout(c, enable_progress)
        c.callbacks += callbacks[:]

        self.state = play(
            c,
            env=self.make_env(),
            worker=self.make_worker(),
            trainer=None,
            workers=None,
        )

        self._after_history()
        return cast(RunStateActor, self.state)

    def update_context_rollout(self, context: Optional[RunContext] = None, enable_progress: bool = False):
        c = self.context if context is None else context
        c.play_mode = "rollout"
        # --- stop config
        # c.max_episodes = max_episodes
        # c.timeout = timeout
        # c.max_steps = max_steps
        c.max_train_count = 0
        # c.max_memory = max_memory
        # --- play config
        # c.players = players
        # c.shuffle_player = shuffle_player
        c.disable_trainer = True
        # --- train option
        # c.train_interval = train_interval
        # c.train_repeat = train_repeat
        # --- play info
        c.distributed = False
        c.training = True
        c.train_only = False
        c.rollout = True
        # --- render
        c.env_render_mode = ""
        c.rl_render_mode = ""

        if enable_progress:
            self.apply_progress(c.callbacks, apply_eval=False)
        self.apply_checkpoint(c.callbacks)
        self._apply_history(c.callbacks)
        self.apply_mlflow(c.callbacks)
        return c

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
        c = self.context.copy()
        c.timeout = timeout
        c.max_train_count = max_train_count
        self.update_context_train_only(c, enable_progress)
        c.callbacks += callbacks[:]

        from srl.base.run.core_train_only import RunStateTrainer, play_trainer_only

        self.state = play_trainer_only(
            c,
            trainer=self.make_trainer(),
        )

        self._after_history()
        return cast(RunStateTrainer, self.state)

    def update_context_train_only(self, context: Optional[RunContext] = None, enable_progress: bool = False):
        c = self.context if context is None else context
        c.play_mode = "train_only"
        # --- stop config
        c.max_episodes = 0
        # c.timeout = timeout
        c.max_steps = 0
        # c.max_train_count = max_train_count
        c.max_memory = 0
        # --- play config
        # c.players = players
        c.shuffle_player = False
        c.disable_trainer = False
        # --- play info
        c.distributed = False
        c.training = True
        c.train_only = True
        c.rollout = False
        # --- render
        c.env_render_mode = ""
        c.rl_render_mode = ""

        if enable_progress:
            self.apply_progress(c.callbacks, apply_eval=True)
        self.apply_checkpoint(c.callbacks)
        self._apply_history(c.callbacks)
        self.apply_mlflow(c.callbacks)
        return c

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
        # return memory
        return_memory_data: bool = False,
        return_memory_timeout: int = 60 * 60 * 1,
        # start data
        initial_parameter_sharing: bool = True,
        initial_memory_sharing: bool = False,
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
        used_context: Optional[RunContext] = None,
    ):
        """multiprocessingを使用した分散学習による学習を実施します。"""
        if used_context is None:
            c = self.context.copy()
            c.actor_num = actor_num
            c.actor_devices = actor_devices
            c.timeout = timeout
            c.max_train_count = max_train_count
            c.players = players
            c.shuffle_player = shuffle_player
            self.update_context_train_mp(c, enable_progress)
            c.callbacks += callbacks[:]
        else:
            c = used_context

        # mp前にsetupを保証する
        if not self.rl_config.is_setup():
            self.rl_config.setup(self.make_env())

        # ---
        if self.rl_config.get_framework() == "tensorflow":
            os.environ["SRL_TF_GPU_INITIALIZE_DEVICES"] = "1"

        params_dat = self._parameter_dat
        if (params_dat is None) and initial_parameter_sharing:
            params_dat = self.make_parameter().backup(serialized=True)

        memory_dat = self._memory_dat
        if (memory_dat is None) and initial_memory_sharing:
            memory_dat = self.make_memory().backup(compress=True)

        if enable_mp_memory:
            from srl.base.run.play_mp_memory import MpConfig, train

            self._parameter_dat, self._memory_dat = train(
                MpConfig(
                    c,
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
        else:
            from srl.base.run.play_mp import MpConfig, train

            self._parameter_dat, self._memory_dat = train(
                MpConfig(
                    c,
                    queue_capacity=queue_capacity,
                    trainer_parameter_send_interval=trainer_parameter_send_interval,
                    actor_parameter_sync_interval=actor_parameter_sync_interval,
                    return_memory_data=return_memory_data,
                    return_memory_timeout=return_memory_timeout,
                ),
                params_dat,
                memory_dat,
            )

        self._after_history()

    def update_context_train_mp(self, context: Optional[RunContext] = None, enable_progress: bool = False):
        c = self.context if context is None else context
        # --- mp config
        # c.actor_num = actor_num
        # c.actor_devices = actor_devices
        c.play_mode = "train_mp"
        # --- stop config
        c.max_episodes = 0
        # c.timeout = timeout
        c.max_steps = 0
        # c.max_train_count = max_train_count
        c.max_memory = 0
        # --- play config
        # c.players = players
        # c.shuffle_player = shuffle_player
        c.disable_trainer = False
        # --- train option
        # c.train_interval = train_interval
        # c.train_repeat = train_repeat
        # --- play info
        c.distributed = True
        c.training = True
        c.train_only = False
        c.rollout = False
        # --- render
        c.env_render_mode = ""
        c.rl_render_mode = ""

        if enable_progress:
            self.apply_progress(c.callbacks, apply_eval=True)
        self.apply_checkpoint(c.callbacks)
        self._apply_history(c.callbacks)
        self.apply_mlflow(c.callbacks)
        return c

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
        actor_devices: Union[str, List[str]] = "AUTO",
        initial_parameter_sharing: bool = True,
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
        used_context: Optional[RunContext] = None,
    ):
        if used_context is None:
            c = self.context.copy()
            c.actor_num = actor_num
            c.actor_devices = actor_devices
            c.timeout = timeout
            c.max_train_count = max_train_count
            c.players = players
            c.shuffle_player = shuffle_player
            self.update_context_train_distribution(c, enable_progress)
        else:
            c = used_context

        # mp前にsetupを保証する
        if not self.rl_config.is_setup():
            self.rl_config.setup(self.make_env())

        # --- parameter
        params_dat = self._parameter_dat
        if (params_dat is None) and initial_parameter_sharing:
            params_dat = self.make_parameter().backup(serialized=True)

        # --- distrubution callback
        from srl.runner.distribution.callback import DistributionCallback

        callbacks_dist: List[DistributionCallback] = []
        callbacks_run: List[RunCallback] = []
        for _c in callbacks:
            if issubclass(_c.__class__, DistributionCallback):
                callbacks_dist.append(cast(DistributionCallback, c))
            else:
                callbacks_run.append(cast(RunCallback, c))
        if len(callbacks_run) > 0:
            logger.warning(f"It must be possible to read it with pickle at the distribution destination. {callbacks}")
        c.callbacks += callbacks_run

        # --- create task
        from srl.runner.distribution.server_manager import TaskConfig, TaskManager

        task_manager = TaskManager(redis_params, "client")
        task_manager.create_task(
            TaskConfig(
                c,
                queue_capacity=queue_capacity,
                trainer_parameter_send_interval=trainer_parameter_send_interval,
                actor_parameter_sync_interval=actor_parameter_sync_interval,
            ),
            params_dat,
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

    def update_context_train_distribution(self, context: Optional[RunContext] = None, enable_progress: bool = False):
        c = self.context if context is None else context
        # --- mp config
        # c.actor_num = actor_num
        # c.actor_devices = actor_devices
        c.play_mode = "train_distribution"
        # --- stop config
        c.max_episodes = 0
        # c.timeout = timeout
        c.max_steps = 0
        # c.max_train_count = max_train_count
        c.max_memory = 0
        # --- play config
        # c.players = players
        # c.shuffle_player = shuffle_player
        c.disable_trainer = False
        # --- play info
        c.distributed = True
        c.training = True
        c.train_only = False
        c.rollout = False
        # --- render
        c.env_render_mode = ""
        c.rl_render_mode = ""

        if enable_progress:
            self.apply_progress(c.callbacks, apply_eval=False)
        return c

    def train_distribution_start(
        self,
        redis_params: "RedisParameters",
        # mp
        actor_num: int = 1,
        queue_capacity: int = 1000,
        trainer_parameter_send_interval: int = 1,
        actor_parameter_sync_interval: int = 1,
        actor_devices: Union[str, List[str]] = "AUTO",
        initial_parameter_sharing: bool = True,
        # --- stop config
        timeout: float = -1,
        max_train_count: int = -1,
        # --- play config
        players: PlayersType = [],
        shuffle_player: bool = True,
        # --- other
        enable_progress: bool = True,
        callbacks: List[RunCallback] = [],
        used_context: Optional[RunContext] = None,
    ):
        if used_context is None:
            c = self.context.copy()
            c.actor_num = actor_num
            c.actor_devices = actor_devices
            c.timeout = timeout
            c.max_train_count = max_train_count
            c.players = players
            c.shuffle_player = shuffle_player
            self.update_context_train_distribution(c, enable_progress)
        else:
            c = used_context
        if len(callbacks) > 0:
            logger.warning(f"It must be possible to read it with pickle at the distribution destination. {callbacks}")
        c.callbacks += callbacks[:]

        # mp前にsetupを保証する
        if not self.rl_config.is_setup():
            self.rl_config.setup(self.make_env())

        # --- parameter
        params_dat = self._parameter_dat
        if (params_dat is None) and initial_parameter_sharing:
            params_dat = self.make_parameter().backup(serialized=True)

        # --- create task
        from srl.runner.distribution.server_manager import TaskConfig, TaskManager

        task_manager = TaskManager(redis_params, "client")
        task_manager.create_task(
            TaskConfig(
                c,
                queue_capacity=queue_capacity,
                trainer_parameter_send_interval=trainer_parameter_send_interval,
                actor_parameter_sync_interval=actor_parameter_sync_interval,
            ),
            params_dat,
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
            players (PlayerTypes, optional): 二人以上の環境で他プレイヤーのアルゴリズム
            shuffle_player (bool, optional): playersをシャッフルするかどうか. Defaults to True.
            enable_progress (bool, optional): 進捗を表示するか. Defaults to True.
            callbacks (List[RunCallback], optional): callbacks. Defaults to [].

        Returns:
            Union[List[float], List[List[float]]]: プレイヤー数が1人なら Lost[float]、複数なら List[List[float]]] を返します。
        """
        c = self.context.copy()
        c.max_episodes = max_episodes
        c.timeout = timeout
        c.max_steps = max_steps
        c.players = players
        c.shuffle_player = shuffle_player
        self.update_context_evaluate(c, enable_progress)
        c.callbacks += callbacks[:]

        self.state = play(
            c,
            env=self.make_env(),
            worker=self.make_worker(),
            trainer=None,
            workers=None,
        )

        state = cast(RunStateActor, self.state)
        if self.env.player_num == 1:
            return [r[0] for r in state.episode_rewards_list]
        else:
            return state.episode_rewards_list

    def update_context_evaluate(self, context: Optional[RunContext] = None, enable_progress: bool = False):
        c = self.context if context is None else context
        c.play_mode = "evaluate"
        # --- stop config
        # c.max_episodes = max_episodes
        # c.timeout = timeout
        # c.max_steps = max_steps
        c.max_train_count = 0
        c.max_memory = 0
        # --- play config
        # c.players = players
        # c.shuffle_player = shuffle_player
        c.disable_trainer = True
        # --- play info
        c.distributed = False
        c.training = False
        c.train_only = False
        c.rollout = False
        # --- render
        c.env_render_mode = ""
        c.rl_render_mode = ""

        if enable_progress:
            self.apply_progress(c.callbacks, apply_eval=False)
        return c

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
        c = self.context.copy()
        c.play_mode = "render_terminal"
        # --- stop config
        c.max_episodes = 1
        c.timeout = timeout
        c.max_steps = max_steps
        c.max_train_count = 0
        c.max_memory = 0
        # --- play config
        c.players = players
        c.shuffle_player = False
        c.disable_trainer = True
        # --- play info
        c.distributed = False
        c.training = training_flag
        c.train_only = False
        c.rollout = False

        # --- rendering ---
        from srl.runner.callbacks.rendering import Rendering

        c.callbacks.append(
            Rendering(
                mode="terminal",
                kwargs=render_kwargs,
                step_stop=step_stop,
                render_skip_step=render_skip_step,
            )
        )
        # -----------------

        c.callbacks += callbacks[:]

        if training_flag:
            params_dat = self.parameter.backup()

        self.state = play(
            c,
            env=self.make_env(),
            worker=self.make_worker(),
            trainer=self.make_trainer() if training_flag else None,
            workers=None,
        )

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
        c = self.context.copy()
        c.play_mode = "render_window"
        # --- stop config
        c.max_episodes = 1
        c.timeout = timeout
        c.max_steps = max_steps
        c.max_train_count = 0
        c.max_memory = 0
        # --- play config
        c.players = players
        c.shuffle_player = False
        c.disable_trainer = True
        # --- play info
        c.distributed = False
        c.training = training_flag
        c.train_only = False
        c.rollout = False
        # --- render_modeはRendering側で設定
        # c.env_render_mode = ""
        # c.rl_render_mode = ""

        # -----------------
        from srl.runner.callbacks.rendering import Rendering

        rendering = Rendering(
            mode="window",
            kwargs=render_kwargs,
            step_stop=False,
            render_interval=render_interval,
            render_skip_step=render_skip_step,
        )
        c.callbacks.append(rendering)
        # -----------------

        if enable_progress:
            self.apply_progress(c.callbacks, apply_eval=False)

        c.callbacks += callbacks[:]

        if training_flag:
            params_dat = self.parameter.backup()

        self.state = play(
            c,
            env=self.make_env(),
            worker=self.make_worker(),
            trainer=self.make_trainer() if self.context.training else None,
            workers=None,
        )

        if training_flag:
            self.parameter.restore(params_dat)

    def _run_render(
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
        c = self.context.copy()
        c.play_mode = "run_render"
        # --- stop config
        c.max_episodes = 1
        c.timeout = timeout
        c.max_steps = max_steps
        c.max_train_count = 0
        c.max_memory = 0
        # --- play config
        c.players = players
        c.shuffle_player = False
        c.disable_trainer = True
        # --- play info
        c.distributed = False
        c.training = training_flag
        c.train_only = False
        c.rollout = False
        # --- render_modeはRendering側で設定
        # c.env_render_mode = ""
        # c.rl_render_mode = ""

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
        c.callbacks.append(rendering)
        # -----------------

        if enable_progress:
            self.apply_progress(c.callbacks, apply_eval=False)
        c.callbacks += callbacks[:]

        if training_flag:
            params_dat = self.parameter.backup()

        self.state = play(
            c,
            env=self.make_env(),
            worker=self.make_worker(),
            trainer=self.make_trainer() if self.context.training else None,
            workers=None,
        )

        if training_flag:
            self.parameter.restore(params_dat)

        if self.context.run_name != "eval":
            state = cast(RunStateActor, self.state)
            logger.info(f"render step: {state.total_step}, reward: {state.episode_rewards_list}")
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
        kwargs = {
            k: v  #
            for k, v in locals().items()
            if k not in ["self", "path", "render_interval", "render_scale"]
        }
        render = self._run_render(**kwargs)
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
        kwargs = {
            k: v  #
            for k, v in locals().items()
            if k not in ["self", "path", "render_interval", "render_scale", "codec"]
        }
        render = self._run_render(**kwargs)
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
        kwargs = {
            k: v  #
            for k, v in locals().items()
            if k not in ["self", "render_interval", "render_scale"]
        }
        render = self._run_render(**kwargs)
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
        c = self.context.copy()
        c.play_mode = "replay_window"
        # --- stop config
        c.max_episodes = 1
        c.timeout = timeout
        c.max_steps = max_steps
        c.max_train_count = 0
        c.max_memory = 0
        # --- play config
        c.players = players
        c.shuffle_player = False
        c.disable_trainer = True
        # --- play info
        c.distributed = False
        c.training = training_flag
        c.train_only = False
        c.rollout = False
        # --- render_modeはRePlayableGame側で設定
        # self.context.env_render_mode = ""
        # self.context.rl_render_mode = ""

        if enable_progress:
            self.apply_progress(c.callbacks, apply_eval=False)
        c.callbacks += callbacks[:]

        if training_flag:
            params_dat = self.parameter.backup()

        from srl.runner.game_windows.replay_window import RePlayableGame

        self.context.callbacks = callbacks
        window = RePlayableGame(
            c,
            env=self.make_env(),
            worker=self.make_worker(),
            trainer=self.make_trainer() if self.context.training else None,
            render_player=render_player,
            print_state=print_state,
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
        c = self.context.copy()
        c.play_mode = "play_terminal"
        # --- stop config
        c.max_episodes = 1
        c.timeout = timeout
        c.max_steps = max_steps
        c.max_train_count = 0
        c.max_memory = 0
        # --- play config
        c.players = players
        c.shuffle_player = False
        c.disable_trainer = True
        # --- play info
        c.distributed = False
        c.training = enable_memory or training_flag
        c.train_only = False
        c.rollout = enable_memory
        # --- render_modeはRendering側で設定
        # self.context.env_render_mode = ""
        # self.context.rl_render_mode = ""

        # --- rendering ---
        from srl.runner.callbacks.rendering import Rendering

        rendering = Rendering(
            mode="terminal",
            kwargs=render_kwargs,
            step_stop=step_stop,
            render_skip_step=render_skip_step,
            render_env=False,
        )
        c.callbacks.append(rendering)

        # -----------------
        from srl.runner.callbacks.manual_play_callback import ManualPlayCallback

        c.callbacks.append(ManualPlayCallback(self.make_env(), action_division_num))
        # -----------------

        c.callbacks += callbacks[:]

        if training_flag:
            params_dat = self.parameter.backup()

        self.state = play(
            c,
            env=self.make_env(),
            worker=self.make_worker(),
            trainer=None,
            workers=None,
        )

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
        c = self.context.copy()
        c.play_mode = "play_window"
        # --- stop config
        c.max_episodes = -1
        c.timeout = timeout
        c.max_steps = max_steps
        c.max_train_count = 0
        c.max_memory = 0
        # --- play config
        c.players = players
        c.shuffle_player = False
        c.disable_trainer = True
        # --- play info
        c.distributed = False
        c.training = enable_memory or training_flag
        c.train_only = False
        c.rollout = enable_memory
        # --- render_modeはPlayableGame側で設定
        # self.context.env_render_mode = ""
        # self.context.rl_render_mode = ""

        c.callbacks += callbacks[:]

        if training_flag:
            params_dat = self.parameter.backup()

        from srl.utils.common import is_packages_installed

        error_text = "This run requires installation of 'PIL', 'pygame'. "
        error_text += "(pip install pillow pygame)"
        assert is_packages_installed(["PIL", "pygame"]), error_text

        from srl.runner.game_windows.playable_game import PlayableGame

        game = PlayableGame(
            c,
            env=self.make_env(),
            worker=self.make_worker(),
            trainer=self.make_trainer() if self.context.training else None,
            view_state=view_state,
            action_division_num=action_division_num,
            key_bind=key_bind,
            _is_test=_is_test,
        )
        game.play()

        if training_flag:
            self.parameter.restore(params_dat)

    # --------------------------------------------
    # utils
    # --------------------------------------------
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

    def summary(self, show_changed_only: bool = False, show_parameter: bool = False):
        if self.rl_config.is_setup():
            self.rl_config.setup(self.make_env())
        self.context.summary(show_changed_only)
        if show_parameter:
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


def load_runner(path_or_cfg_dict: Union[dict, Any, str]) -> Runner:
    context = RunContext.load(path_or_cfg_dict)
    return Runner(context=context)
