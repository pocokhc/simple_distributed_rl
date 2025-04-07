import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Union, cast

from srl.base.define import PlayerType, RenderModes
from srl.base.run.callback import RunCallback
from srl.runner.runner_base import RunnerBase

if TYPE_CHECKING:
    from srl.runner.distribution.callback import DistributionCallback
    from srl.runner.distribution.connectors.redis_ import RedisParameters

logger = logging.getLogger(__name__)


@dataclass()
class RunnerFacadeDistribution(RunnerBase):
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
        players: List[PlayerType] = [],
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
        self.context.rendering = False
        self.context.render_mode = RenderModes.none

        if enable_progress:
            self.apply_progress(callbacks_run, enable_eval=False)

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
        players: List[PlayerType] = [],
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
        self.context.rendering = False
        self.context.render_mode = RenderModes.none

        if enable_progress:
            self.apply_progress(callbacks, enable_eval=False)

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
