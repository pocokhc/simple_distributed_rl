import logging
import queue
import time
import traceback
from typing import List, Optional, cast

import srl
from srl.base.rl.base import IRLMemoryWorker, RLMemory, RLParameter
from srl.base.run.context import RunNameTypes
from srl.runner.callback import Callback
from srl.runner.distribution.manager import DistributedManager
from srl.runner.runner import TaskConfig

logger = logging.getLogger(__name__)


class _ActorRLMemory(IRLMemoryWorker):
    def __init__(self, manager: DistributedManager, task_id: str, manager_queue_capacity: int):
        self.manager = manager
        self.task_id = task_id
        self.manager_queue_capacity = manager_queue_capacity
        self.count = 0
        self.q = queue.Queue()

    def add(self, *args) -> None:
        while True:
            qsize = self.manager.memory_size(self.task_id)
            if 0 <= qsize < self.manager_queue_capacity:
                self.manager.memory_add(self.task_id, args)
                self.count += 1
                break

            # keepalive
            if self.manager.keepalive(self.task_id):
                if self.manager.task_get_status(self.task_id) == "END":
                    break

            time.sleep(1)

    def length(self) -> int:
        return self.count


class _ActorInterrupt(Callback):
    def __init__(
        self,
        manager: DistributedManager,
        task_id: str,
        actor_idx: int,
        parameter: RLParameter,
        actor_parameter_sync_interval_by_step: int,
    ) -> None:
        self.manager = manager
        self.task_id = task_id
        self.actor_idx = actor_idx
        self.parameter = parameter
        self.actor_parameter_sync_interval_by_step = actor_parameter_sync_interval_by_step
        self.sync_step = 0

    def on_episodes_begin(self, runner: srl.Runner):
        runner.state.sync_actor = 0

    def on_step_end(self, runner: srl.Runner) -> bool:
        # --- sync params
        self.sync_step += 1
        if self.sync_step >= self.actor_parameter_sync_interval_by_step:
            body = self.manager.parameter_read(self.task_id)
            if body is not None:
                self.parameter.restore(body)
                self.sync_step = 0
                runner.state.sync_actor += 1

        # --- keepalive
        if self.manager.keepalive(self.task_id):
            self.manager.task_set_actor(self.task_id, self.actor_idx, "episode", runner.state.episode_count)

            if self.manager.task_get_status(self.task_id) == "END":
                return True
        return False


def _run_actor(manager: DistributedManager, task_id: str, task_config: TaskConfig, actor_id: int):
    task_config.context.run_name = RunNameTypes.actor
    task_config.context.actor_id = actor_id
    logger.info(f"actor_id={actor_id}")

    # --- runner
    runner = srl.Runner(
        task_config.context.env_config,
        task_config.context.rl_config,
        task_config.config,
        task_config.context,
    )

    # --- memory
    memory = _ActorRLMemory(manager, task_id, task_config.config.mp_queue_capacity)

    # --- parameter
    parameter = runner.make_parameter()
    params = manager.parameter_read(task_id)
    if params is None:
        logger.warning("Missing initial parameters")
    else:
        parameter.restore(params)

    # --- play
    runner.context.callbacks.append(
        _ActorInterrupt(
            manager,
            task_id,
            actor_id,
            parameter,
            runner.context.actor_parameter_sync_interval_by_step,
        )
    )
    runner.context.disable_trainer = True
    runner.context.max_episodes = -1
    runner.context.max_memory = -1
    runner.context.max_steps = -1
    runner.context.max_train_count = -1
    runner.context.timeout = -1
    runner.core_play(trainer_only=False, memory=cast(RLMemory, memory))


class ActorServerCallback:
    def on_polling(self) -> Optional[bool]:
        """If return is True, it will end intermediate stop."""
        return False


def run_forever(
    host: str,
    port: int = 6379,
    redis_kwargs: dict = {},
    keepalive_interval: int = 10,
    callbacks: List[ActorServerCallback] = [],
):
    manager = DistributedManager(host, port, redis_kwargs, keepalive_interval)
    manager.server_ping()
    manager.set_user("actor")

    print(f"wait actor: {manager.uid}")
    while True:
        try:
            time.sleep(1)

            # --- callback
            _stop_flags = [c.on_polling() for c in callbacks]
            if True in _stop_flags:
                break

            # --- board check
            task_id, task_config, actor_id = manager.task_assign_by_my_id()

            # --- actor loop
            if task_config is not None:
                try:
                    logger.info(f"actor{manager.uid} start")
                    _run_actor(manager, task_id, task_config, actor_id)
                    logger.info(f"actor{manager.uid} end")
                finally:
                    print(f"wait actor: {manager.uid}")

        except Exception:
            logger.warning(traceback.format_exc())
