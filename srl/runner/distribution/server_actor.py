import datetime
import logging
import queue
import random
import threading
import time
import traceback
from typing import List, Optional, cast

import srl
from srl.base.define import RLMemoryTypes
from srl.base.exception import DistributionError
from srl.base.rl.base import IRLMemoryWorker, RLMemory, RLParameter
from srl.base.run.callback import RunCallback
from srl.base.run.context import RunContext, RunNameTypes
from srl.base.run.core_play import RunStateActor
from srl.runner.distribution.callback import ActorServerCallback
from srl.runner.distribution.connectors.parameters import RedisParameters
from srl.runner.distribution.interface import IMemoryServerParameters
from srl.runner.distribution.server_manager import ServerManager
from srl.runner.distribution.task_manager import TaskManager, TaskManagerParams

logger = logging.getLogger(__name__)


# ------------------------------------------
# thread(step | add)
# ------------------------------------------
class _ShareData:
    def __init__(self):
        self.sync_count = 0
        self.step = 0
        self.q_send_count = 0
        self.q_recv_count = 0
        self.end_signal = False
        self.th_error = ""


class _ActorRLMemoryThread(IRLMemoryWorker):
    def __init__(
        self,
        share_q: queue.Queue,
        share_data: _ShareData,
        dist_queue_capacity: int,
    ):
        self.q = share_q
        self.share_data = share_data
        self.dist_queue_capacity = dist_queue_capacity

    @property
    def memory_type(self) -> RLMemoryTypes:
        return RLMemoryTypes.NONE

    def add(self, *args) -> None:
        t0 = time.time()
        while True:
            if self.q.qsize() < self.dist_queue_capacity / 2:
                self.q.put(args)
                break

            if self.share_data.end_signal:
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
        return self.share_data.q_send_count


class _ActorInterruptThread(RunCallback):
    def __init__(
        self,
        share_data: _ShareData,
        memory_ps: threading.Thread,
        parameter_ps: threading.Thread,
    ) -> None:
        self.share_data = share_data
        self.memory_ps = memory_ps
        self.parameter_ps = parameter_ps

    def on_episodes_begin(self, context: RunContext, state: RunStateActor):
        state.sync_actor = 0
        self.share_data.sync_count = 0

    def on_step_end(self, context: RunContext, state: RunStateActor) -> bool:
        state.sync_actor = self.share_data.sync_count
        self.share_data.step = state.total_step
        state.actor_send_q = state.memory.length()
        if not self.memory_ps.is_alive():
            self.share_data.end_signal = True
        if not self.parameter_ps.is_alive():
            self.share_data.end_signal = True
        return self.share_data.end_signal


def _memory_communicate(
    manager_copy_args,
    share_q: queue.Queue,
    share_data: _ShareData,
    dist_queue_capacity: int,
    actor_num: int,
    actor_idx: int,
):
    try:
        manager = ServerManager._copy(*manager_copy_args)
        memory_sender = manager.get_memory_sender()

        q_send_count = 0
        while not share_data.end_signal:
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
                    remote_qsize = qsize - share_data.q_recv_count

                if remote_qsize >= dist_queue_capacity / 2:
                    time.sleep(1)
                    continue

                memory_sender.memory_add(share_q.get(timeout=1))
                q_send_count += 1
                share_data.q_send_count = q_send_count
            except Exception as e:
                logger.error(f"Memory send error: {e}")

    except Exception:
        share_data.th_error = traceback.format_exc()
    finally:
        share_data.end_signal = True
        logger.info(f"actor{actor_idx} memory thread end.")


def _parameter_communicate(
    manager_copy_args,
    parameter: RLParameter,
    share_data: _ShareData,
    actor_parameter_sync_interval: int,
    actor_idx: int,
):
    try:
        manager = ServerManager._copy(*manager_copy_args)
        task_manager = manager.get_task_manager()
        parameter_reader = manager.get_parameter_reader()

        keepalive_t0 = 0
        sync_parameter_t0 = time.time()
        while not share_data.end_signal:
            # --- sync parameter
            if time.time() - sync_parameter_t0 > actor_parameter_sync_interval:
                sync_parameter_t0 = time.time()

                params = parameter_reader.parameter_read()
                if params is not None:
                    parameter.restore(params, from_cpu=True)
                    share_data.sync_count += 1

            # --- q_recv_count
            q_recv_count = task_manager.get_trainer("q_recv_count")
            share_data.q_recv_count = 0 if q_recv_count == "" else int(q_recv_count)

            # --- task check
            if time.time() - keepalive_t0 > task_manager.params.keepalive_interval:
                keepalive_t0 = time.time()
                _keepalive(task_manager, actor_idx, share_data.step, share_data.q_send_count)
                if task_manager.is_finished():
                    break

            time.sleep(1)
        _keepalive(task_manager, actor_idx, share_data.step, share_data.q_send_count)

    except Exception:
        share_data.th_error = traceback.format_exc()
    finally:
        share_data.end_signal = True
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

    @property
    def memory_type(self) -> RLMemoryTypes:
        return RLMemoryTypes.NONE

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
        actor_id: int,
        actor_parameter_sync_interval: int,
    ) -> None:
        self.task_manager = manager.get_task_manager()
        self.parameter_reader = manager.get_parameter_reader()
        self.keepalive_interval = self.task_manager.params.keepalive_interval
        self.actor_id = actor_id

        self.parameter = parameter
        self.actor_parameter_sync_interval = actor_parameter_sync_interval
        self.t0 = time.time()

    def on_episodes_begin(self, context: RunContext, state: RunStateActor):
        state.sync_actor = 0
        self._keepalive_t0 = 0

    def on_step_end(self, context: RunContext, state: RunStateActor) -> bool:
        # --- sync params
        if time.time() - self.t0 > self.actor_parameter_sync_interval:
            self.t0 = time.time()

            body = self.parameter_reader.parameter_read()
            if body is not None:
                self.parameter.restore(body, from_cpu=True)
                state.sync_actor += 1

        state.actor_send_q = state.memory.length()

        # --- keepalive
        if time.time() - self._keepalive_t0 < self.keepalive_interval:
            _keepalive(self.task_manager, self.actor_id, state.total_step, state.memory.length())
            if self.task_manager.is_finished():
                return True
        return False

    def on_episodes_end(self, context: RunContext, state: RunStateActor) -> None:
        _keepalive(self.task_manager, self.actor_id, state.total_step, state.memory.length())


def _run_actor(manager: ServerManager, runner: srl.Runner):
    task_manager = manager.get_task_manager()
    parameter = runner.make_parameter(is_load=False)

    _t = task_manager.get_config()
    callbacks = [] if _t is None else _t.callbacks

    # --- thread
    if runner.config.dist_enable_actor_thread:
        share_data = _ShareData()
        share_q = queue.Queue()
        _manager_copy_args = manager._copy_args()
        memory_ps = threading.Thread(
            target=_memory_communicate,
            args=(
                _manager_copy_args,
                share_q,
                share_data,
                runner.config.dist_queue_capacity,
                task_manager.get_actor_num(),
                runner.context.actor_id,
            ),
        )
        parameter_ps = threading.Thread(
            target=_parameter_communicate,
            args=(
                _manager_copy_args,
                parameter,
                share_data,
                runner.config.actor_parameter_sync_interval,
                runner.context.actor_id,
            ),
        )
        memory_ps.start()
        parameter_ps.start()
        memory = _ActorRLMemoryThread(
            share_q,
            share_data,
            runner.config.dist_queue_capacity,
        )
        callbacks.append(_ActorInterruptThread(share_data, memory_ps, parameter_ps))
    else:
        memory_ps = None
        parameter_ps = None
        share_data = None
        memory = _ActorRLMemoryNoThread(manager, runner.config.dist_queue_capacity)
        callbacks.append(
            _ActorInterruptNoThread(
                manager,
                parameter,
                runner.context.actor_id,
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
            parameter=parameter,
            memory=cast(RLMemory, memory),
            trainer=None,
            workers=None,
            callbacks=callbacks,
            enable_generator=False,
        )
    except DistributionError:
        raise
    finally:
        if memory_ps is not None:
            assert share_data is not None
            assert memory_ps is not None
            assert parameter_ps is not None
            share_data.end_signal = True
            memory_ps.join(timeout=10)
            parameter_ps.join(timeout=10)
            if share_data.th_error != "":
                raise ValueError(share_data.th_error)


def _keepalive(task_manager: TaskManager, actor_id: int, step: int, q_send_count: int):
    # --- 2重にアサインされていないかチェック
    aid = task_manager.get_actor(actor_id, "id")
    if aid != task_manager.params.uid:
        # アサインされていたらランダム秒まって止める
        s = f"Another actor has been assigned. my:{task_manager.params.uid}, another: {aid}"
        task_manager.add_log(s)
        time.sleep(random.randint(0, 5))
        raise DistributionError(s)

    # update
    task_manager.set_actor(actor_id, "step", str(step))
    task_manager.set_actor(actor_id, "q_send_count", str(q_send_count))
    task_manager.set_actor(actor_id, "update_time", task_manager.get_now_str())


def _task_assign(task_manager: TaskManager) -> Optional[srl.Runner]:
    if task_manager.get_status() != "ACTIVE":
        return None

    # --- queue が setup されてから実行する
    if not task_manager.is_setup_memory():
        return None

    # --- runnerが作れるか
    runner = task_manager.create_runner(read_parameter=True)
    if runner is None:
        return None

    # --- env が動かせるか
    runner.make_env()

    now_utc = datetime.datetime.now(datetime.timezone.utc)

    # --- 自分のIDが既にあれば続きから入る
    for i in range(task_manager.get_actor_num()):
        _aid = task_manager.get_actor(i, "id")
        if _aid == task_manager.params.uid:
            runner.context.run_name = RunNameTypes.actor
            runner.context.actor_id = i
            task_manager.set_actor(i, "update_time", task_manager.get_now_str())
            task_manager.add_log(f"Actor{i} reassigned({task_manager.params.uid})")
            task_manager.check_version()
            return runner

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

        runner.context.run_name = RunNameTypes.actor
        runner.context.actor_id = i
        task_manager.set_actor(i, "id", task_manager.params.uid)
        task_manager.set_actor(i, "update_time", task_manager.get_now_str())
        task_manager.add_log(f"Actor{i} assigned({task_manager.params.uid})")
        task_manager.check_version()
        return runner

    return None


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
    used_device_tf, used_device_torch = srl.Runner.setup_device(framework, device)
    task_manager_params = TaskManagerParams(
        "actor",
        keepalive_interval,
        keepalive_threshold,
        framework=framework,
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
            task_manager.reset()
            runner = _task_assign(task_manager)
            if runner is not None:
                print(f"actor{uid} start, actor_id={runner.context.actor_id}")
                logger.info(f"actor{uid} start, actor_id={runner.context.actor_id}")
                _run_actor(manager, runner)
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
                time.sleep(10)
                print(f"wait actor: {uid}")
                logger.info(f"wait actor: {uid}")
