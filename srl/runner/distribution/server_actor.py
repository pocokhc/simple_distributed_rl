import logging
import queue
import threading
import time
import traceback
from typing import List, Optional, cast

import srl
from srl.base.exception import DistributionError
from srl.base.rl.base import IRLMemoryWorker, RLMemory, RLParameter
from srl.base.run.context import RunNameTypes
from srl.runner.callback import Callback
from srl.runner.distribution.callback import ActorServerCallback
from srl.runner.distribution.connectors.imemory import IServerParameters
from srl.runner.distribution.connectors.parameters import RedisParameters
from srl.runner.distribution.manager import DistributedManager
from srl.runner.runner import Runner, TaskConfig

logger = logging.getLogger(__name__)

# remote_memoryは接続が切れたら再接続
# redisは接続が切れたら終了


# ------------------------------------------
# thread(step | add)
# ------------------------------------------
class _ActorRLMemoryThread(IRLMemoryWorker):
    def __init__(
        self,
        share_dict: dict,
        dist_queue_capacity: int,
        manager: DistributedManager,
        actor_idx: int,
    ):
        self.share_dict = share_dict
        self.dist_queue_capacity = dist_queue_capacity
        self.q = queue.Queue()
        self.remote_memory = manager.create_memory_connector()
        self.manager = manager
        self.actor_num = self.manager.task_get_actor_num()
        self.actor_idx = actor_idx

    def add(self, *args) -> None:
        t0 = time.time()
        while True:
            # --- server check
            remote_qsize = -1
            if not self.remote_memory.is_connected:
                self.remote_memory.ping()
            if self.remote_memory.is_connected:
                remote_qsize = self.remote_memory.memory_size()

            # remote_qsizeが取得できない場合は受信と送信から予測
            if remote_qsize < 0:
                # 他のactorのsendを取得
                qsize = 0
                for idx in range(self.actor_num):
                    if idx == self.actor_idx:
                        continue
                    _q = self.manager.task_get_actor(idx, "q_send_count")
                    qsize += 0 if _q == "" else int(_q)
                qsize += self.share_dict["q_send_count"]
                remote_qsize = qsize - self.share_dict["q_recv_count"]

            # --- qが一定以下のみ送信
            if remote_qsize + self.q.qsize() < self.dist_queue_capacity:
                self.q.put(args)
                break

            if self.share_dict["end_signal"]:
                break

            if time.time() - t0 > 10:
                t0 = time.time()
                s = "capacity over, wait:"
                s += f"local {self.q.qsize()}"
                s += f", remote_qsize {remote_qsize}"
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
                continue

            if not remote_memory.is_connected:
                time.sleep(1)
                if not remote_memory.ping():
                    continue

            try:
                remote_memory.memory_add(memory.q.get(timeout=1))
                q_send_count += 1
                share_dict["q_send_count"] = q_send_count
            except Exception as e:
                logger.error(f"Memory send error: {e}")

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
    def __init__(self, manager: DistributedManager, dist_queue_capacity: int, actor_idx: int):
        self.manager = manager
        self.remote_memory = manager.create_memory_connector()
        self.dist_queue_capacity = dist_queue_capacity
        self.q_send_count = 0
        self.q = queue.Queue()
        self.actor_num = self.manager.task_get_actor_num()
        self.actor_idx = actor_idx

    def add(self, *args) -> None:
        t0 = time.time()
        while True:
            # --- server check
            remote_qsize = -1
            if not self.remote_memory.is_connected:
                self.remote_memory.ping()
            if self.remote_memory.is_connected:
                remote_qsize = self.remote_memory.memory_size()

            # remote_qsizeが取得できない場合は受信と送信から予測
            if remote_qsize < 0:
                # 他のactorのsendを取得
                qsize = 0
                for idx in range(self.actor_num):
                    if idx == self.actor_idx:
                        continue
                    _q = self.manager.task_get_actor(idx, "q_send_count")
                    qsize += 0 if _q == "" else int(_q)
                qsize += self.q_send_count
                q_recv_count = self.manager.task_get_trainer("q_recv_count")
                q_recv_count = 0 if q_recv_count == "" else int(q_recv_count)
                remote_qsize = qsize - q_recv_count

            # --- qが一定以下のみ送信
            if remote_qsize < self.dist_queue_capacity:
                try:
                    self.remote_memory.memory_add(args)
                    self.q_send_count += 1
                except Exception as e:
                    logger.error(e)
                break

            # keepalive
            if self.manager.keepalive():
                if self.manager.task_is_dead():
                    raise DistributionError("task is dead")

            if time.time() - t0 > 10:
                t0 = time.time()
                s = "capacity over, wait:"
                s += f"local {self.q.qsize()}"
                s += f", remote_qsize {remote_qsize}"
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
        memory = _ActorRLMemoryThread(share_dict, task_config.config.dist_queue_capacity, manager, actor_idx)
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
        memory = _ActorRLMemoryManager(manager, task_config.config.dist_queue_capacity, actor_idx)
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
    except DistributionError:
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
):
    used_device_tf, used_device_torch = Runner.setup_device(framework, device)

    manager = DistributedManager(redis_parameter, memory_parameter)
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

            if not manager.ping():
                logger.info("Server connect fail.")
                time.sleep(10)
                continue

            # --- queue が setup されてから実行する
            if not manager.task_is_setup_queue():
                continue

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
