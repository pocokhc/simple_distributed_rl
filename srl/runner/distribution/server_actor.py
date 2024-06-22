import datetime
import logging
import queue
import random
import threading
import time
import traceback
from typing import List, Optional

from srl.base.context import RunContext, RunNameTypes
from srl.base.env.env_run import EnvRun
from srl.base.exception import DistributionError
from srl.base.rl.memory import IRLMemoryWorker
from srl.base.rl.parameter import RLParameter
from srl.base.run import core_play
from srl.base.run.callback import RunCallback
from srl.base.run.core_play import RunStateActor
from srl.base.system.device import setup_device
from srl.runner.distribution.callback import ActorServerCallback
from srl.runner.distribution.connectors.parameters import RedisParameters
from srl.runner.distribution.interface import IMemoryServerParameters
from srl.runner.distribution.server_manager import ServerManager
from srl.runner.distribution.task_manager import TaskConfig, TaskManager, TaskManagerParams

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
        base_memory: IRLMemoryWorker,
    ):
        self.q = share_q
        self.share_data = share_data
        self.dist_queue_capacity = dist_queue_capacity
        self.base_memory = base_memory

    def add(self, *args, serialized: bool = False) -> None:
        t0 = time.time()
        while True:
            if self.q.qsize() < self.dist_queue_capacity / 2:
                raw = self.base_memory.serialize_add_args(*args)
                self.q.put(raw)
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

    def serialize_add_args(self, *args):
        raise NotImplementedError("Unused")


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

    def on_episodes_begin(self, context: RunContext, state: RunStateActor, **kwargs):
        state.sync_actor = 0
        self.share_data.sync_count = 0

    def on_step_end(self, context: RunContext, state: RunStateActor, **kwargs) -> bool:
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
    def __init__(self, manager: ServerManager, dist_queue_capacity: int, base_memory: IRLMemoryWorker):
        self.task_manager = manager.get_task_manager()
        self.remote_memory = manager.get_memory_sender()

        self.dist_queue_capacity = dist_queue_capacity
        self.q_send_count = 0
        self.q = queue.Queue()
        self.actor_num = self.task_manager.get_actor_num()
        self.base_memory = base_memory

        self.keepalive_t0 = 0

    def add(self, *args, serialized: bool = False) -> None:
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
                        raw = self.base_memory.serialize_add_args(*args)
                        self.remote_memory.memory_add(raw)
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

    def serialize_add_args(self, *args):
        raise NotImplementedError("Unused")


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

    def on_episodes_begin(self, context: RunContext, state: RunStateActor, **kwargs):
        state.sync_actor = 0
        self._keepalive_t0 = 0

    def on_step_end(self, context: RunContext, state: RunStateActor, **kwargs) -> bool:
        # --- sync params
        if time.time() - self.t0 > self.actor_parameter_sync_interval:
            self.t0 = time.time()

            params = self.parameter_reader.parameter_read()
            if params is not None:
                self.parameter.restore(params, from_cpu=True)
                state.sync_actor += 1

        state.actor_send_q = state.memory.length()

        # --- keepalive
        if time.time() - self._keepalive_t0 < self.keepalive_interval:
            _keepalive(self.task_manager, self.actor_id, state.total_step, state.memory.length())
            if self.task_manager.is_finished():
                return True
        return False

    def on_episodes_end(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        _keepalive(self.task_manager, self.actor_id, state.total_step, state.memory.length())


def _run_actor(
    manager: ServerManager,
    task_config: TaskConfig,
    env: EnvRun,
    parameter: RLParameter,
    actor_id: int,
):
    task_manager = manager.get_task_manager()
    callbacks = task_config.get_run_callback()

    # --- parameter
    params = manager.get_parameter_reader().parameter_read()
    if params is not None:
        parameter.restore(params, from_cpu=True)

    # --- thread
    if task_config.enable_actor_thread:
        share_data = _ShareData()
        share_q = queue.Queue()
        _manager_copy_args = manager._copy_args()
        memory_ps = threading.Thread(
            target=_memory_communicate,
            args=(
                _manager_copy_args,
                share_q,
                share_data,
                task_config.queue_capacity,
                task_manager.get_actor_num(),
                task_config.context.actor_id,
            ),
        )
        parameter_ps = threading.Thread(
            target=_parameter_communicate,
            args=(
                _manager_copy_args,
                parameter,
                share_data,
                task_config.actor_parameter_sync_interval,
                task_config.context.actor_id,
            ),
        )
        memory_ps.start()
        parameter_ps.start()
        memory = _ActorRLMemoryThread(
            share_q,
            share_data,
            task_config.queue_capacity,
            task_config.context.rl_config.make_memory(is_load=False),
        )
        callbacks.append(_ActorInterruptThread(share_data, memory_ps, parameter_ps))
    else:
        memory_ps = None
        parameter_ps = None
        share_data = None
        memory = _ActorRLMemoryNoThread(
            manager,
            task_config.queue_capacity,
            task_config.context.rl_config.make_memory(is_load=False),
        )
        callbacks.append(
            _ActorInterruptNoThread(
                manager,
                parameter,
                task_config.context.actor_id,
                task_config.actor_parameter_sync_interval,
            )
        )

    # --- play
    context = task_config.context
    context.training = True
    context.disable_trainer = True
    context.enable_train_thread = False
    # context.max_episodes = -1
    # context.max_memory = -1
    # context.max_steps = -1
    # context.max_train_count = -1
    # context.timeout = -1
    workers, main_worker_idx = context.rl_config.make_workers(context.players, env, parameter, memory)
    try:
        core_play.play(
            context=context,
            env=env,
            workers=workers,
            main_worker_idx=main_worker_idx,
            trainer=None,
            callbacks=callbacks,
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


def _task_assign(task_manager: TaskManager):
    if task_manager.get_status() != "ACTIVE":
        return None

    # --- queue が setup されてから実行する
    if not task_manager.is_setup_memory():
        return None

    # --- create/check
    task_config = task_manager.get_config()
    if task_config is None:
        return None
    env = task_config.context.env_config.make()
    parameter = task_config.context.rl_config.make_parameter(is_load=False)
    task_manager.read_parameter(parameter)

    now_utc = datetime.datetime.now(datetime.timezone.utc)

    # --- 自分のIDが既にあれば続きから入る
    for i in range(task_manager.get_actor_num()):
        _aid = task_manager.get_actor(i, "id")
        if _aid == task_manager.params.uid:
            task_config.context.run_name = RunNameTypes.actor
            task_config.context.actor_id = i
            task_manager.set_actor(i, "update_time", task_manager.get_now_str())
            task_manager.add_log(f"Actor{i} reassigned({task_manager.params.uid})")
            task_manager.check_version()
            return task_config, env, parameter, i

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

        task_config.context.run_name = RunNameTypes.actor
        task_config.context.actor_id = i
        task_manager.set_actor(i, "id", task_manager.params.uid)
        task_manager.set_actor(i, "update_time", task_manager.get_now_str())
        task_manager.add_log(f"Actor{i} assigned({task_manager.params.uid})")
        task_manager.check_version()
        return task_config, env, parameter, i

    return None


def run_forever(
    redis_params: RedisParameters,
    memory_params: Optional[IMemoryServerParameters] = None,
    keepalive_interval: int = 10,
    keepalive_threshold: int = 101,
    callbacks: List[ActorServerCallback] = [],
    framework: str = "tensorflow",
    device: str = "CPU",
    set_CUDA_VISIBLE_DEVICES_if_CPU: bool = True,
    tf_enable_memory_growth: bool = True,
    run_once: bool = False,
):
    used_device_tf, used_device_torch = setup_device(
        framework,
        device,
        set_CUDA_VISIBLE_DEVICES_if_CPU,
        tf_enable_memory_growth,
    )
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
            run_params = _task_assign(task_manager)
            if run_params is not None:
                actor_id = run_params[-1]
                print(f"actor{uid} start, actor_id={actor_id}")
                logger.info(f"actor{uid} start, actor_id={actor_id}")
                _run_actor(manager, *run_params)
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
