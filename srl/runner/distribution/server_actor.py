import logging
import queue
import threading
import time
import traceback
from typing import List, Optional, cast

import srl
from srl.base.rl.base import IRLMemoryWorker, RLMemory, RLParameter
from srl.base.run.context import RunNameTypes
from srl.runner.callback import Callback
from srl.runner.distribution.callback import ActorServerCallback
from srl.runner.distribution.connectors.imemory import IServerParameters
from srl.runner.distribution.connectors.parameters import RedisParameters
from srl.runner.distribution.manager import DistributedManager
from srl.runner.runner import Runner, TaskConfig

logger = logging.getLogger(__name__)


class ActorException(Exception):
    pass


# ------------------------------------------
# thread(step | add)
# ------------------------------------------
class _ActorRLMemoryThread(IRLMemoryWorker):
    def __init__(
        self,
        share_dict: dict,
        dist_queue_capacity: int,
        manager: DistributedManager,
    ):
        self.share_dict = share_dict
        self.dist_queue_capacity = dist_queue_capacity
        self.q = queue.Queue()
        self.remote_memory = manager.create_memory_connector()

    def add(self, *args) -> None:
        t0 = time.time()
        while True:
            _is_send_q = True

            # --- 受信と送信でN以上差があれば待機
            diff_qsize = self.share_dict["q_send_count"] - self.share_dict["q_recv_count"]
            if diff_qsize > self.dist_queue_capacity:
                _is_send_q = False

            # --- qsizeがN以上なら待機、sizeはリアルタイムで取る
            remote_memory_qsize = self.remote_memory.memory_size()
            qsize = remote_memory_qsize + self.q.qsize()
            if qsize >= self.dist_queue_capacity:
                _is_send_q = False

            if _is_send_q:
                self.q.put(args)
                break
            if self.share_dict["end_signal"]:
                break
            if time.time() - t0 > 10:
                t0 = time.time()
                s = "capacity over, wait:"
                s += f"local {self.q.qsize()}"
                s += f", server {remote_memory_qsize}"
                s += f", send/recv {diff_qsize}"
                print(s)
                logger.info(s)
                break
            time.sleep(1)

    def length(self) -> int:
        return self.q.qsize()


class _ActorInterruptThread(Callback):
    def __init__(self, memory_ps: threading.Thread, parameter_ps: threading.Thread, share_dict: dict) -> None:
        self.memory_ps = memory_ps
        self.parameter_ps = parameter_ps
        self.share_dict = share_dict

    def on_episodes_begin(self, runner: srl.Runner):
        runner.state.sync_actor = 0
        self.share_dict["sync_count"] = 0

    def on_step_end(self, runner: srl.Runner) -> bool:
        runner.state.sync_actor = self.share_dict["sync_count"]
        if not self.memory_ps.is_alive():
            self.share_dict["end_signal"] = True
        if not self.parameter_ps.is_alive():
            self.share_dict["end_signal"] = True
        return self.share_dict["end_signal"]

    def on_episode_end(self, runner: srl.Runner):
        self.share_dict["episode_count"] = runner.state.episode_count


def _memory_communicate(
    manager_args,
    actor_idx: int,
    memory: _ActorRLMemoryThread,
    share_dict: dict,
):
    try:
        manager = DistributedManager.create(*manager_args)
        remote_memory = manager.create_memory_connector()

        q_send_count = 0
        while not share_dict["end_signal"]:
            if memory.q.empty():
                time.sleep(0.1)
            else:
                remote_memory.memory_add(memory.q.get(timeout=1))
                q_send_count += 1
                share_dict["q_send_count"] = q_send_count

    except Exception:
        share_dict["th_error"] = traceback.format_exc()
    finally:
        share_dict["end_signal"] = True
        logger.info(f"actor{actor_idx} memory thread end.")


def _parameter_communicate(
    manager_args,
    actor_idx: int,
    parameter: RLParameter,
    share_dict: dict,
    actor_parameter_sync_interval: int,
):
    try:
        manager = DistributedManager.create(*manager_args)

        t0 = time.time()
        while not share_dict["end_signal"]:
            # --- sync parameter
            if time.time() - t0 > actor_parameter_sync_interval:
                t0 = time.time()

                params = manager.parameter_read()
                if params is not None:
                    parameter.restore(params, from_cpu=True)
                    share_dict["sync_count"] += 1

            q_recv_count = manager.task_get_trainer("q_recv_count")
            share_dict["q_recv_count"] = 0 if q_recv_count == "" else int(q_recv_count)

            # --- keepalive
            if manager.keepalive():
                manager.task_set_actor(actor_idx, "episode", str(share_dict["episode_count"]))
                manager.task_set_actor(actor_idx, "q_send_count", str(share_dict["q_send_count"]))
                if manager.task_is_dead():
                    logger.info("task is dead")
                    break

            time.sleep(1)

        manager.keepalive(do_now=True)
        manager.task_set_actor(actor_idx, "episode", str(share_dict["episode_count"]))
        manager.task_set_actor(actor_idx, "q_send_count", str(share_dict["q_send_count"]))

    except Exception:
        share_dict["th_error"] = traceback.format_exc()
    finally:
        share_dict["end_signal"] = True
        logger.info(f"actor{actor_idx} parameter thread end.")


# ------------------------------------------
# no thread(step -> add)
# ------------------------------------------
class _ActorRLMemoryManager(IRLMemoryWorker):
    def __init__(self, manager: DistributedManager, dist_queue_capacity: int):
        self.manager = manager
        self.remote_memory = manager.create_memory_connector()
        self.dist_queue_capacity = dist_queue_capacity
        self.q_send_count = 0
        self.q = queue.Queue()

    def add(self, *args) -> None:
        t0 = time.time()
        while True:
            _is_send_q = True

            # --- 受信と送信でN以上差があれば待機
            q_recv_count = self.manager.task_get_trainer("q_recv_count")
            q_recv_count = 0 if q_recv_count == "" else int(q_recv_count)
            if self.q_send_count - q_recv_count > self.dist_queue_capacity:
                _is_send_q = False

            # --- qsizeがN以上なら待機
            qsize = self.remote_memory.memory_size()
            if qsize >= self.dist_queue_capacity:
                _is_send_q = False

            if _is_send_q:
                self.remote_memory.memory_add(args)
                self.q_send_count += 1
                break

            # keepalive
            if self.manager.keepalive():
                if self.manager.task_is_dead():
                    raise ActorException("task is dead")

            if time.time() - t0 > 10:
                t0 = time.time()
                s = "capacity over, wait:"
                s += f"local {qsize}"
                s += f", server {self.dist_queue_capacity}"
                s += f", send/recv {self.q_send_count - q_recv_count}"
                print(s)
                logger.info(s)
                break

            time.sleep(1)

    def length(self) -> int:
        return self.q_send_count


class _ActorInterruptManager(Callback):
    def __init__(
        self,
        manager: DistributedManager,
        actor_idx: int,
        parameter: RLParameter,
        actor_parameter_sync_interval: int,
    ) -> None:
        self.manager = manager
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

            body = self.manager.parameter_read()
            if body is not None:
                self.parameter.restore(body, from_cpu=True)
                runner.state.sync_actor += 1

        # --- keepalive
        if self.manager.keepalive():
            assert runner.state.memory is not None
            self.manager.task_set_actor(self.actor_idx, "episode", str(runner.state.episode_count))
            self.manager.task_set_actor(self.actor_idx, "q_send_count", str(runner.state.memory.length()))
            if self.manager.task_is_dead():
                logger.info("task is dead")
                return True
        return False

    def on_episodes_end(self, runner: Runner) -> None:
        assert runner.state.memory is not None
        self.manager.keepalive(do_now=True)
        self.manager.task_set_actor(self.actor_idx, "episode", str(runner.state.episode_count))
        self.manager.task_set_actor(self.actor_idx, "q_send_count", str(runner.state.memory.length()))


# --------------------------------
# main
# --------------------------------
def _run_actor(manager: DistributedManager, task_config: TaskConfig, actor_idx: int):
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
    params = manager.parameter_read()
    if params is None:
        logger.warning("Missing initial parameters")
    else:
        parameter.restore(params, from_cpu=True)

    # --- thread
    if task_config.config.dist_enable_actor_thread:
        share_dict = {
            "sync_count": 0,
            "episode_count": 0,
            "q_send_count": 0,
            "q_recv_count": 0,
            "end_signal": False,
            "th_error": "",
        }
        memory = _ActorRLMemoryThread(share_dict, task_config.config.dist_queue_capacity, manager)
        memory_ps = threading.Thread(
            target=_memory_communicate,
            args=(
                manager.create_args(),
                actor_idx,
                memory,
                share_dict,
            ),
        )
        parameter_ps = threading.Thread(
            target=_parameter_communicate,
            args=(
                manager.create_args(),
                actor_idx,
                parameter,
                share_dict,
                task_config.config.actor_parameter_sync_interval,
            ),
        )
        memory_ps.start()
        parameter_ps.start()
        runner.context.callbacks.append(_ActorInterruptThread(memory_ps, parameter_ps, share_dict))
    else:
        memory_ps = None
        parameter_ps = None
        share_dict = {}
        memory = _ActorRLMemoryManager(manager, task_config.config.dist_queue_capacity)
        runner.context.callbacks.append(
            _ActorInterruptManager(
                manager,
                actor_idx,
                parameter,
                runner.config.actor_parameter_sync_interval,
            )
        )

    # --- play
    runner.context.disable_trainer = True
    # runner.context.max_episodes = -1
    # runner.context.max_memory = -1
    # runner.context.max_steps = -1
    # runner.context.max_train_count = -1
    # runner.context.timeout = -1
    try:
        runner.core_play(
            trainer_only=False,
            parameter=parameter,
            memory=cast(RLMemory, memory),
            trainer=None,
            workers=None,
        )
    except ActorException:
        raise
    finally:
        if memory_ps is not None:
            share_dict["end_signal"] = True
            assert memory_ps is not None
            assert parameter_ps is not None
            memory_ps.join(timeout=10)
            parameter_ps.join(timeout=10)
            if share_dict["th_error"] != "":
                raise ValueError(share_dict["th_error"])


def run_forever(
    redis_parameter: RedisParameters,
    memory_parameter: Optional[IServerParameters] = None,
    callbacks: List[ActorServerCallback] = [],
    framework: str = "tensorflow",
    device: str = "CPU",
    run_once: bool = False,
    is_remote_memory_purge: bool = True,
):
    used_device_tf, used_device_torch = Runner.setup_device(framework, device)

    manager = DistributedManager(redis_parameter, memory_parameter)
    assert manager.ping()
    manager.set_user("actor")

    print(f"wait actor: {manager.uid}")
    logger.info(f"wait actor: {manager.uid}")
    while True:
        try:
            time.sleep(1)

            # --- callback
            _stop_flags = [c.on_polling() for c in callbacks]
            if True in _stop_flags:
                break

            # --- task check
            is_assigned, actor_id = manager.task_assign_by_my_id()
            if is_assigned:
                print(f"actor{manager.uid} start, actor_id={actor_id}")
                logger.info(f"actor{manager.uid} start, actor_id={actor_id}")
                task_config = manager.task_get_config()
                assert task_config is not None
                task_config.context.create_controller().set_device(used_device_tf, used_device_torch)
                task_config.context.used_device_tf = used_device_tf
                task_config.context.used_device_torch = used_device_torch
                if is_remote_memory_purge:
                    manager.create_memory_connector().memory_purge()
                _run_actor(manager, task_config, actor_id)
                logger.info(f"actor{manager.uid} end")
                if run_once:
                    break
                print(f"wait actor: {manager.uid}")
                logger.info(f"wait actor: {manager.uid}")

        except Exception:
            if run_once:
                raise
            else:
                logger.error(traceback.format_exc())
                print(f"wait actor: {manager.uid}")
                logger.info(f"wait actor: {manager.uid}")
