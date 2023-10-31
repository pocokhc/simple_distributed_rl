import logging
import queue
import threading
import time
import traceback
from typing import List, cast

import srl
from srl.base.rl.base import IRLMemoryWorker, RLMemory, RLParameter
from srl.base.run.context import RunNameTypes
from srl.runner.callback import Callback
from srl.runner.distribution.callback import ActorServerCallback
from srl.runner.distribution.manager import DistributedManager
from srl.runner.runner import Runner, TaskConfig

logger = logging.getLogger(__name__)


class ActorException(Exception):
    pass


# ------------------------------------------
# thread(step | add)
# ------------------------------------------
class _ActorRLMemoryThread(IRLMemoryWorker):
    def __init__(self, share_dict: dict, dist_queue_capacity: int):
        self.share_dict = share_dict
        self.dist_queue_capacity = dist_queue_capacity
        self.q = queue.Queue()

    def add(self, *args) -> None:
        t0 = time.time()
        while True:
            qsize = self.share_dict["qsize"] + self.q.qsize()
            if 0 <= qsize < self.dist_queue_capacity:
                self.q.put(args)
                break
            if self.share_dict["end_signal"]:
                break
            if time.time() - t0 > 10:
                t0 = time.time()
                print(f"capacity over, wait queue: {qsize}/{self.dist_queue_capacity}")
            time.sleep(1)

    def length(self) -> int:
        return self.q.qsize()


class _ActorInterruptThread(Callback):
    def __init__(self, server_ps: threading.Thread, share_dict: dict) -> None:
        self.server_ps = server_ps
        self.share_dict = share_dict

    def on_episodes_begin(self, runner: srl.Runner):
        runner.state.sync_actor = 0
        self.share_dict["sync_count"] = 0

    def on_step_end(self, runner: srl.Runner) -> bool:
        runner.state.sync_actor = self.share_dict["sync_count"]
        if not self.server_ps.is_alive():
            self.share_dict["end_signal"] = True
        return self.share_dict["end_signal"]

    def on_episode_end(self, runner: srl.Runner):
        self.share_dict["episode_count"] = runner.state.episode_count


def _server_communicate(
    manager_args,
    task_id: str,
    actor_idx: int,
    memory: _ActorRLMemoryThread,
    parameter: RLParameter,
    share_dict: dict,
    actor_parameter_sync_interval: int,
):
    try:
        manager = DistributedManager.create(*manager_args)
        t0 = time.time()
        while True:
            # --- send memory
            share_dict["qsize"] = manager.memory_size(task_id)
            if memory.length() > 0:
                manager.memory_add(task_id, memory.q.get())

            # --- sync parameter
            if time.time() - t0 > actor_parameter_sync_interval:
                t0 = time.time()

                params = manager.parameter_read(task_id)
                if params is not None:
                    parameter.restore(params)
                    share_dict["sync_count"] += 1

            # --- keepalive
            if manager.keepalive(task_id):
                manager.task_set_actor(task_id, actor_idx, "episode", str(share_dict["episode_count"]))
                if manager.task_is_dead(task_id):
                    share_dict["end_signal"] = True
                    logger.info(f"task is dead: {task_id}")
                    break

    except Exception:
        logger.info(traceback.format_exc())
    finally:
        share_dict["end_signal"] = True
    logger.info(f"actor{actor_idx} thread end.")


# ------------------------------------------
# no thread(step -> add)
# ------------------------------------------
class _ActorRLMemoryManager(IRLMemoryWorker):
    def __init__(self, manager: DistributedManager, task_id: str, dist_queue_capacity: int):
        self.manager = manager
        self.task_id = task_id
        self.dist_queue_capacity = dist_queue_capacity
        self.count = 0
        self.q = queue.Queue()

    def add(self, *args) -> None:
        while True:
            qsize = self.manager.memory_size(self.task_id)
            if 0 <= qsize < self.dist_queue_capacity:
                self.manager.memory_add(self.task_id, args)
                self.count += 1
                break

            # keepalive
            if self.manager.keepalive(self.task_id):
                if self.manager.task_is_dead(self.task_id):
                    raise ActorException(f"task is dead: {self.task_id}")

                print(f"capacity over, wait queue: {qsize}/{self.dist_queue_capacity}")
            time.sleep(1)

    def length(self) -> int:
        return self.count


class _ActorInterruptManager(Callback):
    def __init__(
        self,
        manager: DistributedManager,
        task_id: str,
        actor_idx: int,
        parameter: RLParameter,
        actor_parameter_sync_interval: int,
    ) -> None:
        self.manager = manager
        self.task_id = task_id
        self.actor_idx = actor_idx
        self.parameter = parameter
        self.actor_parameter_sync_interval = actor_parameter_sync_interval
        self.t0 = time.time()

    def on_episodes_begin(self, runner: srl.Runner):
        runner.state.sync_actor = 0

    def on_step_end(self, runner: srl.Runner) -> bool:
        # --- sync params
        if time.time() - self.t0 > self.actor_parameter_sync_interval:
            self.t0 = time.time()

            body = self.manager.parameter_read(self.task_id)
            if body is not None:
                self.parameter.restore(body)
                runner.state.sync_actor += 1

        # --- keepalive
        if self.manager.keepalive(self.task_id):
            self.manager.task_set_actor(self.task_id, self.actor_idx, "episode", str(runner.state.episode_count))
            if self.manager.task_is_dead(self.task_id):
                logger.info(f"task is dead: {self.task_id}")
                return True
        return False


# --------------------------------
# main
# --------------------------------
def _run_actor(manager: DistributedManager, task_id: str, task_config: TaskConfig, actor_idx: int):
    task_config.context.run_name = RunNameTypes.actor
    task_config.context.actor_id = actor_idx

    # --- runner
    runner = srl.Runner(
        task_config.context.env_config,
        task_config.context.rl_config,
        task_config.config,
        task_config.context,
    )

    # --- parameter
    parameter = runner.make_parameter()
    params = manager.parameter_read(task_id)
    if params is None:
        logger.warning("Missing initial parameters")
    else:
        parameter.restore(params)

    # --- thread
    if task_config.config.dist_enable_actor_thread:
        share_dict = {
            "sync_count": 0,
            "qsize": 0,
            "episode_count": 0,
            "end_signal": False,
        }
        memory = _ActorRLMemoryThread(share_dict, task_config.config.dist_queue_capacity)
        server_ps = threading.Thread(
            target=_server_communicate,
            args=(
                manager.create_args(),
                task_id,
                actor_idx,
                memory,
                parameter,
                share_dict,
                task_config.config.actor_parameter_sync_interval,
            ),
        )
        server_ps.start()
        runner.context.callbacks.append(_ActorInterruptThread(server_ps, share_dict))
    else:
        memory = _ActorRLMemoryManager(manager, task_id, task_config.config.dist_queue_capacity)
        runner.context.callbacks.append(
            _ActorInterruptManager(
                manager,
                task_id,
                actor_idx,
                parameter,
                runner.config.actor_parameter_sync_interval,
            )
        )

    # --- play
    runner.context.disable_trainer = True
    runner.context.max_episodes = -1
    runner.context.max_memory = -1
    runner.context.max_steps = -1
    runner.context.max_train_count = -1
    runner.context.timeout = -1
    try:
        runner.core_play(
            trainer_only=False,
            parameter=None,
            memory=cast(RLMemory, memory),
            trainer=None,
            workers=None,
        )
    except ActorException as e:
        logger.info(e)


def run_forever(
    host: str,
    redis_kwargs: dict = {},
    keepalive_interval: int = 10,
    callbacks: List[ActorServerCallback] = [],
    framework: str = "tensorflow",
    device: str = "CPU",
):
    used_device_tf, used_device_torch = Runner.setup_device(framework, device)

    manager = DistributedManager(host, redis_kwargs, keepalive_interval)
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

            # --- task check
            task_id, task_config, actor_id = manager.task_assign_by_my_id()
            if task_config is not None:
                try:
                    print(f"actor{manager.uid} start, actor_id={actor_id}")
                    logger.info(f"actor{manager.uid} start, actor_id={actor_id}")
                    task_config = cast(TaskConfig, task_config)
                    task_config.context.create_controller().set_device(used_device_tf, used_device_torch)
                    task_config.context.used_device_tf = used_device_tf
                    task_config.context.used_device_torch = used_device_torch
                    _run_actor(manager, task_id, task_config, actor_id)
                    logger.info(f"actor{manager.uid} end")
                finally:
                    print(f"wait actor: {manager.uid}")

        except Exception:
            logger.warning(traceback.format_exc())
