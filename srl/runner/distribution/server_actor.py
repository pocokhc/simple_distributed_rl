import datetime
import logging
import queue
import random
import threading
import time
import traceback
from typing import List, Optional, Tuple, cast

import srl
from srl.base.exception import DistributionError
from srl.base.rl.base import IRLMemoryWorker, RLMemory, RLParameter
from srl.base.run.callback import RunCallback
from srl.base.run.context import RunContext, RunNameTypes
from srl.base.run.core import RunState
from srl.runner.distribution.callback import ActorServerCallback
from srl.runner.distribution.connectors.parameters import RedisParameters
from srl.runner.distribution.interface import IMemoryServerParameters
from srl.runner.distribution.server_manager import ServerManager
from srl.runner.distribution.task_manager import TaskManager, TaskManagerParams
from srl.runner.runner import Runner

logger = logging.getLogger(__name__)


# ------------------------------------------
# thread(step | add)
# ------------------------------------------
class _ActorRLMemoryThread(IRLMemoryWorker):
    def __init__(
        self,
        share_q: queue.Queue,
        share_dict: dict,
        dist_queue_capacity: int,
    ):
        self.q = share_q
        self.share_dict = share_dict
        self.dist_queue_capacity = dist_queue_capacity

    def add(self, *args) -> None:
        t0 = time.time()
        while True:
            if self.q.qsize() < self.dist_queue_capacity / 2:
                self.q.put(args)
                break

            if self.share_dict["end_signal"]:
                break

            if time.time() - t0 > 10:
                t0 = time.time()
                s = "queue capacity over:"
                s += f"local {self.q.qsize()}"
                print(s)
                logger.info(s)
                break  # for safety

            time.sleep(1)

    def length(self) -> int:
        return self.share_dict["q_send_count"]


class _ActorInterruptThread(RunCallback):
    def __init__(
        self,
        share_dict: dict,
        memory_ps: threading.Thread,
        parameter_ps: threading.Thread,
    ) -> None:
        self.share_dict = share_dict
        self.memory_ps = memory_ps
        self.parameter_ps = parameter_ps

    def on_episodes_begin(self, context: RunContext, state: RunState):
        state.sync_actor = 0
        self.share_dict["sync_count"] = 0

    def on_step_end(self, context: RunContext, state: RunState) -> bool:
        state.sync_actor = self.share_dict["sync_count"]
        if not self.memory_ps.is_alive():
            self.share_dict["end_signal"] = True
        if not self.parameter_ps.is_alive():
            self.share_dict["end_signal"] = True
        return self.share_dict["end_signal"]

    def on_episode_end(self, context: RunContext, state: RunState):
        self.share_dict["episode_count"] = state.episode_count


def _memory_communicate(
    manager_copy_args,
    share_q: queue.Queue,
    share_dict: dict,
    dist_queue_capacity: int,
    actor_num: int,
    actor_idx: int,
):
    try:
        manager = ServerManager._copy(*manager_copy_args)
        memory_sender = manager.get_memory_sender()

        q_send_count = 0
        while not share_dict["end_signal"]:
            if share_q.empty():
                time.sleep(0.1)
                continue
            try:
                if not memory_sender.is_connected:
                    memory_sender.ping()
                if not memory_sender.is_connected:
                    time.sleep(1)
                    continue

                remote_qsize = memory_sender.memory_size()
                if remote_qsize < 0:
                    # remote_qsizeが取得できない場合は受信と送信から予測
                    # 他のactorは概算
                    qsize = q_send_count * actor_num
                    remote_qsize = qsize - share_dict["q_recv_count"]

                if remote_qsize >= dist_queue_capacity / 2:
                    time.sleep(1)
                    continue

                memory_sender.memory_add(share_q.get(timeout=1))
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
    manager_copy_args,
    parameter: RLParameter,
    share_dict: dict,
    actor_parameter_sync_interval: int,
    actor_idx: int,
):
    try:
        manager = ServerManager._copy(*manager_copy_args)
        task_manager = manager.get_task_manager()
        parameter_reader = manager.get_parameter_reader()

        keepalive_t0 = 0
        sync_parameter_t0 = time.time()
        while not share_dict["end_signal"]:
            # --- sync parameter
            if time.time() - sync_parameter_t0 > actor_parameter_sync_interval:
                sync_parameter_t0 = time.time()

                params = parameter_reader.parameter_read()
                if params is not None:
                    parameter.restore(params, from_cpu=True)
                    share_dict["sync_count"] += 1

            # --- q_recv_count
            q_recv_count = task_manager.get_trainer("q_recv_count")
            share_dict["q_recv_count"] = 0 if q_recv_count == "" else int(q_recv_count)

            # --- task check
            if time.time() - keepalive_t0 > task_manager.params.keepalive_interval:
                keepalive_t0 = time.time()
                _keepalive(task_manager, share_dict["episode_count"], share_dict["q_send_count"])
                if task_manager.is_finished():
                    break

            time.sleep(1)
        _keepalive(task_manager, share_dict["episode_count"], share_dict["q_send_count"])

    except Exception:
        share_dict["th_error"] = traceback.format_exc()
    finally:
        share_dict["end_signal"] = True
        logger.info(f"actor{actor_idx} parameter thread end.")


# ------------------------------------------
# no thread(step -> add)
# ------------------------------------------
class _ActorRLMemoryNoThread(IRLMemoryWorker):
    def __init__(self, manager: ServerManager, dist_queue_capacity: int):
        self.task_manager = manager.get_task_manager()
        self.remote_memory = manager.get_memory_sender()

        self.dist_queue_capacity = dist_queue_capacity
        self.q_send_count = 0
        self.q = queue.Queue()
        self.actor_num = self.task_manager.get_actor_num()

        self.keepalive_t0 = 0

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
                    qsize = self.q_send_count * self.actor_num
                    q_recv_count = self.task_manager.get_trainer("q_recv_count")
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

            # --- keepalive
            if time.time() - self.keepalive_t0 > self.task_manager.params.keepalive_interval:
                self.keepalive_t0 = time.time()
                if self.task_manager.is_finished():
                    raise DistributionError("task is finished")

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


class _ActorInterruptNoThread(RunCallback):
    def __init__(
        self,
        manager: ServerManager,
        parameter: RLParameter,
        actor_parameter_sync_interval: int,
    ) -> None:
        self.task_manager = manager.get_task_manager()
        self.parameter_reader = manager.get_parameter_reader()
        self.keepalive_interval = self.task_manager.params.keepalive_interval

        self.parameter = parameter
        self.actor_parameter_sync_interval = actor_parameter_sync_interval
        self.t0 = time.time()

    def on_episodes_begin(self, context: RunContext, state: RunState):
        state.sync_actor = 0
        self._keepalive_t0 = 0

    def on_step_end(self, context: RunContext, state: RunState) -> bool:
        # --- sync params
        if time.time() - self.t0 > self.actor_parameter_sync_interval:
            self.t0 = time.time()

            body = self.parameter_reader.parameter_read()
            if body is not None:
                self.parameter.restore(body, from_cpu=True)
                state.sync_actor += 1

        # --- keepalive
        if time.time() - self._keepalive_t0 < self.keepalive_interval:
            assert state.memory is not None
            _keepalive(self.task_manager, state.episode_count, state.memory.length())
            if self.task_manager.is_finished():
                return True
        return False

    def on_episodes_end(self, context: RunContext, state: RunState) -> None:
        assert state.memory is not None
        _keepalive(self.task_manager, state.episode_count, state.memory.length())


def _run_actor(manager: ServerManager):
    task_manager = manager.get_task_manager()

    task_config = task_manager.get_config()
    assert task_config is not None
    task_config.context.run_name = RunNameTypes.actor
    task_config.context.actor_id = task_manager.params.actor_idx

    # --- runner
    runner = srl.Runner(
        task_config.context.env_config,
        task_config.context.rl_config,
        task_config.config,
        task_config.context,
    )

    # --- parameter
    parameter = runner.make_parameter(is_load=False)
    params = manager.get_parameter_reader().parameter_read()
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
        share_q = queue.Queue()
        _manager_copy_args = manager._copy_args()
        memory_ps = threading.Thread(
            target=_memory_communicate,
            args=(
                _manager_copy_args,
                share_q,
                share_dict,
                task_config.config.dist_queue_capacity,
                task_manager.get_actor_num(),
                task_manager.params.actor_idx,
            ),
        )
        parameter_ps = threading.Thread(
            target=_parameter_communicate,
            args=(
                _manager_copy_args,
                parameter,
                share_dict,
                task_config.config.actor_parameter_sync_interval,
                task_manager.params.actor_idx,
            ),
        )
        memory_ps.start()
        parameter_ps.start()
        memory = _ActorRLMemoryThread(
            share_q,
            share_dict,
            task_config.config.dist_queue_capacity,
        )
        task_config.callbacks.append(_ActorInterruptThread(share_dict, memory_ps, parameter_ps))
    else:
        memory_ps = None
        parameter_ps = None
        share_dict = {}
        memory = _ActorRLMemoryNoThread(manager, task_config.config.dist_queue_capacity)
        task_config.callbacks.append(
            _ActorInterruptNoThread(
                manager,
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
        runner.base_run_play(
            trainer_only=False,
            parameter=parameter,
            memory=cast(RLMemory, memory),
            trainer=None,
            workers=None,
            callbacks=task_config.callbacks,
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


def _keepalive(task_manager: TaskManager, episode: int, q_send_count: int):
    # --- 2重にアサインされていないかチェック
    aid = task_manager.get_actor(task_manager.params.actor_idx, "id")
    if aid != task_manager.params.uid:
        # アサインされていたらランダム秒まって止める
        s = f"Another actor has been assigned. my:{task_manager.params.uid}, another: {aid}"
        task_manager.add_log(s)
        time.sleep(random.randint(0, 5))
        raise DistributionError(s)

    # update
    task_manager.set_actor(task_manager.params.actor_idx, "episode", str(episode))
    task_manager.set_actor(task_manager.params.actor_idx, "q_send_count", str(q_send_count))
    task_manager.set_actor(task_manager.params.actor_idx, "update_time", task_manager.get_now_str())


def _task_assign(task_manager: TaskManager) -> Tuple[bool, int]:
    if task_manager.get_status() != "ACTIVE":
        return False, 0

    # --- queue が setup されてから実行する
    if not task_manager.is_setup_memory():
        return False, 0

    # --- env が動かせるか
    if not task_manager.is_create_env():
        return False, 0

    now_utc = datetime.datetime.now(datetime.timezone.utc)

    # --- 自分のIDが既にあれば続きから入る
    for i in range(task_manager.get_actor_num()):
        _aid = task_manager.get_actor(i, "id")
        if _aid == task_manager.params.uid:
            task_manager.params.actor_idx = i
            task_manager.set_actor(i, "id", task_manager.params.uid)
            task_manager.set_actor(i, "update_time", task_manager.get_now_str())
            task_manager.add_log(f"Actor{i} reassigned({task_manager.params.uid})")
            task_manager.check_version()
            return True, i

    for i in range(task_manager.get_actor_num()):
        _aid = task_manager.get_actor(i, "id")

        # --- healthチェック ---
        _a_elapsed_time = (now_utc - task_manager.get_actor_update_time(i)).total_seconds()
        if _a_elapsed_time > task_manager.params.keepalive_threshold:
            s = f"Actor{i} remove(health time over) {_a_elapsed_time:.1f}s {_aid}"
            task_manager.add_log(s)
            task_manager.set_actor(i, "id", "")
        # ---------------------

        _aid = task_manager.get_actor(i, "id")
        if _aid != "":
            continue

        task_manager.params.actor_idx = i
        task_manager.set_actor(i, "id", task_manager.params.uid)
        task_manager.set_actor(i, "update_time", task_manager.get_now_str())
        task_manager.add_log(f"Actor{i} assigned({task_manager.params.uid})")
        task_manager.check_version()
        return True, i

    return False, 0


def run_forever(
    redis_params: RedisParameters,
    memory_params: Optional[IMemoryServerParameters] = None,
    keepalive_interval: int = 10,
    keepalive_threshold: int = 101,
    callbacks: List[ActorServerCallback] = [],
    framework: str = "tensorflow",
    device: str = "CPU",
    run_once: bool = False,
):
    used_device_tf, used_device_torch = Runner.setup_device(framework, device)
    task_manager_params = TaskManagerParams(
        "actor",
        keepalive_interval,
        keepalive_threshold,
        used_device_tf=used_device_tf,
        used_device_torch=used_device_torch,
    )
    manager = ServerManager(redis_params, memory_params, task_manager_params)
    task_manager = manager.get_task_manager()
    redis_connector = manager.get_redis_connector()
    memory_sender = manager.get_memory_sender()
    uid = task_manager.params.uid

    print(f"wait actor: {uid}")
    logger.info(f"wait actor: {uid}")
    while True:
        try:
            time.sleep(1)

            # --- callback
            _stop_flags = [c.on_polling() for c in callbacks]
            if True in _stop_flags:
                break

            # --- server check
            if not redis_connector.ping():
                logger.info("Redis server connect fail.")
                time.sleep(10)
                continue
            if not memory_sender.ping():
                logger.info("MemorySender server connect fail.")
                time.sleep(10)
                continue

            # --- task check
            is_assigned, actor_id = _task_assign(task_manager)
            if is_assigned:
                print(f"actor{uid} start, actor_id={actor_id}")
                logger.info(f"actor{uid} start, actor_id={actor_id}")
                _run_actor(manager)
                logger.info(f"actor{uid} end")
                if run_once:
                    break
                print(f"wait actor: {uid}")
                logger.info(f"wait actor: {uid}")

        except Exception:
            if run_once:
                raise
            else:
                logger.error(traceback.format_exc())
                print(f"wait actor: {uid}")
                logger.info(f"wait actor: {uid}")
