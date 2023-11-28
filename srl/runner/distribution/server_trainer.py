import datetime
import logging
import queue
import threading
import time
import traceback
from typing import Any, List, Optional, cast

import srl
from srl.base.define import RLMemoryTypes
from srl.base.rl.base import IRLMemoryTrainer, RLMemory, RLParameter
from srl.base.run.callback import TrainerCallback
from srl.base.run.context import RunContext, RunNameTypes
from srl.base.run.core import RunStateTrainer
from srl.runner.distribution.callback import TrainerServerCallback
from srl.runner.distribution.connectors.parameters import RedisParameters
from srl.runner.distribution.interface import IMemoryServerParameters
from srl.runner.distribution.server_manager import ServerManager
from srl.runner.distribution.task_manager import TaskManager, TaskManagerParams
from srl.runner.runner import Runner

logger = logging.getLogger(__name__)


# ---------------------------------------------------
# thread
#  enable_prepare_sample_batch: True
#   (add -> sample, update) | (train)
#
#  enable_prepare_sample_batch: False
#   (add) | (sample -> train -> update)
#
# ---------------------------------------------------
class _ShareData:
    def __init__(self):
        self.sync_count = 0
        self.train_count = 0
        self.q_recv_count = 0
        self.end_signal = False
        self.th_error = ""


class _TrainerRLMemoryThreadPrepareBatch(IRLMemoryTrainer):
    def __init__(self, base_memory: RLMemory, batch_size: int, share_data: _ShareData):
        # recv,warmup両方threadなので、両方待つ場合は待機
        self.base_memory = base_memory
        self.batch_size = batch_size
        self.share_data = share_data
        self.q_batch = queue.Queue()
        self.q_update = queue.Queue()

    @property
    def memory_type(self) -> RLMemoryTypes:
        return RLMemoryTypes.NONE

    def recv(self, dat) -> None:
        if dat is not None:
            self.base_memory.add(*dat)
        if dat is None and self.base_memory.is_warmup_needed():
            time.sleep(0.1)
        if not self.base_memory.is_warmup_needed():
            if self.q_batch.qsize() < 5:
                self.q_batch.put(self.base_memory.sample(self.batch_size, self.share_data.train_count))
        if not self.q_update.empty():
            self.base_memory.update(self.q_update.get())

    def length(self) -> int:
        return self.base_memory.length()

    def is_warmup_needed(self) -> bool:
        return self.q_batch.empty()

    def sample(self, batch_size: int, step: int) -> Any:
        return self.q_batch.get()

    def update(self, memory_update_args: Any) -> None:
        self.q_update.put(memory_update_args)


def _memory_communicate(
    manager_copy_args,
    memory: RLMemory,
    share_data: _ShareData,
    enable_prepare_sample_batch,
):
    try:
        manager = ServerManager._copy(*manager_copy_args)
        memory_receiver = manager.get_memory_receiver()

        q_recv_count = 0

        if enable_prepare_sample_batch:
            # --- 受信できない場合もsampleを作り続ける
            while not share_data.end_signal:
                try:
                    if not memory_receiver.is_connected:
                        memory_receiver.ping()

                    dat = memory_receiver.memory_recv()
                except Exception as e:
                    logger.error(f"Memory recv error: {e}")
                    continue

                cast(_TrainerRLMemoryThreadPrepareBatch, memory).recv(dat)
                if dat is not None:
                    q_recv_count += 1
                    share_data.q_recv_count = q_recv_count
        else:
            # --- 受信できなければサーバ側なので少し待つ
            while not share_data.end_signal:
                if not memory_receiver.is_connected:
                    time.sleep(1)
                    if not memory_receiver.ping():
                        continue

                try:
                    dat = memory_receiver.memory_recv()
                except Exception as e:
                    logger.error(f"Memory recv error: {e}")
                    continue

                if dat is None:
                    time.sleep(0.1)
                else:
                    q_recv_count += 1
                    share_data.q_recv_count = q_recv_count
                    memory.add(*dat)

    except Exception:
        share_data.th_error = traceback.format_exc()
    finally:
        share_data.end_signal = True
        logger.info("trainer memory thread end.")


def _parameter_communicate(
    manager_copy_args,
    parameter: RLParameter,
    share_data: _ShareData,
    trainer_parameter_send_interval: int,
):
    try:
        manager = ServerManager._copy(*manager_copy_args)
        task_manager = manager.get_task_manager()
        parameter_writer = manager.get_parameter_writer()

        keepalive_t0 = 0
        start_train_count = task_manager.get_train_count()
        while not share_data.end_signal:
            time.sleep(trainer_parameter_send_interval)

            # --- sync parameter
            params = parameter.backup(to_cpu=True)
            if params is not None:
                parameter_writer.parameter_update(params)
                share_data.sync_count += 1

            # --- keepalive
            task_manager.set_trainer("q_recv_count", str(share_data.q_recv_count))
            if time.time() - keepalive_t0 > task_manager.params.keepalive_interval:
                keepalive_t0 = time.time()
                _keepalive(task_manager, start_train_count, share_data.train_count)
                if task_manager.is_finished():
                    break

        task_manager.set_trainer("q_recv_count", str(share_data.q_recv_count))
        _keepalive(task_manager, start_train_count, share_data.train_count)

    except Exception:
        share_data.th_error = traceback.format_exc()
    finally:
        share_data.end_signal = True
        logger.info("trainer parameter thread end.")


class _TrainerInterruptThread(TrainerCallback):
    def __init__(self, memory_ps: threading.Thread, parameter_ps: threading.Thread, share_data: _ShareData) -> None:
        self.memory_ps = memory_ps
        self.parameter_ps = parameter_ps
        self.share_data = share_data

    def on_trainer_loop(self, context: RunContext, state: RunStateTrainer) -> bool:
        if not state.is_step_trained:
            # warmupなら待機
            time.sleep(1)

        state.trainer_recv_q = self.share_data.q_recv_count
        self.share_data.train_count = state.trainer.get_train_count()
        state.sync_trainer = self.share_data.sync_count
        if not self.memory_ps.is_alive():
            self.share_data.end_signal = True
        if not self.parameter_ps.is_alive():
            self.share_data.end_signal = True
        return self.share_data.end_signal


# ------------------------------------------
# no thread(add -> sample -> train -> update)
# ------------------------------------------
class _TrainerInterruptNoThread(TrainerCallback):
    def __init__(
        self,
        manager: ServerManager,
        trainer_parameter_send_interval: int,
    ) -> None:
        self.memory_receiver = manager.get_memory_receiver()
        self.parameter_writer = manager.get_parameter_writer()
        self.task_manager = manager.get_task_manager()
        self.trainer_parameter_send_interval = trainer_parameter_send_interval
        self.q_recv_count = 0

        self.keepalive_t0 = 0
        self.sync_parameter_t0 = time.time()
        self.start_train_count = self.task_manager.get_train_count()

    def on_trainer_loop(self, context: RunContext, state: RunStateTrainer) -> bool:
        state.memory = cast(RLMemory, state.memory)

        # --- recv memory
        if not self.memory_receiver.is_connected:
            self.memory_receiver.ping()
        if self.memory_receiver.is_connected:
            dat = self.memory_receiver.memory_recv()
            if dat is not None:
                self.q_recv_count += 1
                state.memory.add(*dat)
                state.trainer_recv_q = self.q_recv_count
        else:
            dat = None

        # no warmupとmemory emptyなら待つ
        if dat is None and state.memory.is_warmup_needed():
            time.sleep(1)

        # --- sync parameter
        if state.is_step_trained:
            if time.time() - self.sync_parameter_t0 > self.trainer_parameter_send_interval:
                self.sync_parameter_t0 = time.time()

                params = state.parameter.backup(to_cpu=True)
                if params is not None:
                    self.parameter_writer.parameter_update(params)
                    state.sync_trainer += 1

        # --- keepalive
        if time.time() - self.keepalive_t0 > self.task_manager.params.keepalive_interval:
            self.keepalive_t0 = time.time()
            self.task_manager.set_trainer("q_recv_count", str(self.q_recv_count))
            _keepalive(
                self.task_manager,
                self.start_train_count,
                state.trainer.get_train_count(),
            )
            if self.task_manager.is_finished():
                return True

        return False

    def on_trainer_end(self, context: RunContext, state: RunStateTrainer):
        self.task_manager.set_trainer("q_recv_count", str(self.q_recv_count))
        _keepalive(
            self.task_manager,
            self.start_train_count,
            state.trainer.get_train_count(),
        )


def _run_trainer(manager: ServerManager, runner: srl.Runner):
    task_manager = manager.get_task_manager()
    parameter_writer = manager.get_parameter_writer()

    _t = task_manager.get_config()
    callbacks = [] if _t is None else _t.callbacks

    # --- parameter
    parameter = runner.make_parameter(is_load=False)
    params = manager.get_parameter_reader().parameter_read()
    if params is None:
        logger.warning("Missing initial parameters")
    else:
        parameter.restore(params, from_cpu=True)

    # --- memory
    memory = runner.make_memory(is_load=False)

    memory_ps = None
    parameter_ps = None
    share_data = None
    try:
        # --- thread
        if runner.config.dist_enable_trainer_thread:
            share_data = _ShareData()
            if runner.config.dist_enable_prepare_sample_batch:
                batch_size = getattr(runner.context.rl_config, "batch_size", 1)
                memory = _TrainerRLMemoryThreadPrepareBatch(memory, batch_size, share_data)

            _manager_copy_args = manager._copy_args()
            memory_ps = threading.Thread(
                target=_memory_communicate,
                args=(
                    _manager_copy_args,
                    memory,
                    share_data,
                    runner.config.dist_enable_prepare_sample_batch,
                ),
            )
            parameter_ps = threading.Thread(
                target=_parameter_communicate,
                args=(
                    _manager_copy_args,
                    parameter,
                    share_data,
                    runner.config.trainer_parameter_send_interval,
                ),
            )
            memory_ps.start()
            parameter_ps.start()
            callbacks.append(_TrainerInterruptThread(memory_ps, parameter_ps, share_data))
        else:
            callbacks.append(
                _TrainerInterruptNoThread(
                    manager,
                    runner.config.trainer_parameter_send_interval,
                )
            )

        # --- play
        runner.base_run_play_trainer_only(
            parameter=parameter,
            memory=cast(RLMemory, memory),
            trainer=None,
            callbacks=callbacks,
        )
    except Exception:
        raise
    finally:
        # --- last params
        params = parameter.backup(to_cpu=True)
        if params is not None:
            parameter_writer.parameter_update(params)

        task_manager.finished("trainer end")

        if memory_ps is not None:
            assert share_data is not None
            assert memory_ps is not None
            assert parameter_ps is not None
            share_data.end_signal = True
            memory_ps.join(timeout=10)
            parameter_ps.join(timeout=10)
            if share_data.th_error != "":
                raise ValueError(share_data.th_error)


def _keepalive(task_manager: TaskManager, start_train_count: int, train_count: int):
    task_manager.set_train_count(start_train_count, train_count)
    task_manager.set_trainer("train", str(train_count))
    task_manager.set_trainer("update_time", task_manager.get_now_str())


def _task_assign(task_manager: TaskManager) -> Optional[srl.Runner]:
    if task_manager.get_status() != "ACTIVE":
        return None

    # --- runnerが作れるか
    runner = task_manager.create_runner(read_parameter=True)
    if runner is None:
        return None
    runner.context.run_name = RunNameTypes.trainer

    # --- アサイン時にhealthチェック ---
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    _t_elapsed_time = (now_utc - task_manager.get_trainer_update_time()).total_seconds()
    if _t_elapsed_time > task_manager.params.keepalive_threshold:
        _tid = task_manager.get_trainer("id")
        s = f"Trainer remove(health time over) {_t_elapsed_time:.1f}s {_tid}"
        task_manager.add_log(s)
        task_manager.set_trainer("id", "")
    # ----------------------------------

    _tid = task_manager.get_trainer("id")
    if _tid != "":
        return None

    task_manager.set_trainer("id", task_manager.params.uid)
    task_manager.set_trainer("update_time", task_manager.get_now_str())
    task_manager.add_log(f"Trainer assigned({task_manager.params.uid})")
    task_manager.check_version()
    return runner


def run_forever(
    redis_params: RedisParameters,
    memory_params: Optional[IMemoryServerParameters] = None,
    keepalive_interval: int = 10,
    keepalive_threshold: int = 101,
    callbacks: List[TrainerServerCallback] = [],
    framework: str = "tensorflow",
    device: str = "AUTO",
    run_once: bool = False,
    is_remote_memory_purge: bool = True,
):
    used_device_tf, used_device_torch = Runner.setup_device(framework, device)
    task_manager_params = TaskManagerParams(
        "trainer",
        keepalive_interval,
        keepalive_threshold,
        used_device_tf=used_device_tf,
        used_device_torch=used_device_torch,
    )
    manager = ServerManager(redis_params, memory_params, task_manager_params)
    task_manager = manager.get_task_manager()
    redis_connector = manager.get_redis_connector()
    memory_receiver = manager.get_memory_receiver()
    uid = task_manager.params.uid

    print(f"wait trainer: {uid}")
    logger.info(f"wait trainer: {uid}")
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
            if not memory_receiver.ping():
                logger.info("MemoryReceiver server connect fail.")
                time.sleep(10)
                continue

            # --- task check
            task_manager.reset()
            runner = _task_assign(task_manager)
            if runner is not None:
                print(f"train start: {uid}")
                logger.info(f"train start: {uid}")
                task_manager.setup_memory(memory_receiver, is_remote_memory_purge)
                _run_trainer(manager, runner)
                logger.info(f"train end: {uid}")
                if run_once:
                    break
                print(f"wait trainer: {uid}")
                logger.info(f"wait trainer: {uid}")

        except Exception:
            if run_once:
                raise
            else:
                logger.error(traceback.format_exc())
                time.sleep(10)
                print(f"wait trainer: {uid}")
                logger.info(f"wait trainer: {uid}")
