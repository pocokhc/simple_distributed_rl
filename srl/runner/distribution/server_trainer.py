import datetime
import logging
import threading
import time
import traceback
from typing import List, Optional, cast

from srl.base.context import RunContext, RunNameTypes
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.run import core_train_only
from srl.base.run.callback import RunCallback
from srl.base.run.core_train_only import RunStateTrainer
from srl.base.system.device import setup_device
from srl.runner.distribution.callback import TrainerServerCallback
from srl.runner.distribution.connectors.parameters import RedisParameters
from srl.runner.distribution.interface import IMemoryServerParameters
from srl.runner.distribution.server_manager import ServerManager
from srl.runner.distribution.task_manager import TaskConfig, TaskManager, TaskManagerParams

logger = logging.getLogger(__name__)


# ---------------------------------------------------
# thread (add) | (sample -> train -> update)
# ---------------------------------------------------
class _ShareData:
    def __init__(self):
        self.sync_count = 0
        self.train_count = 0
        self.q_recv_count = 0
        self.end_signal = False
        self.th_error = ""


def _memory_communicate(
    manager_copy_args,
    memory: RLMemory,
    share_data: _ShareData,
):
    try:
        manager = ServerManager._copy(*manager_copy_args)
        memory_receiver = manager.get_memory_receiver()
        worker_funcs = {k: v[0] for k, v in memory.get_worker_funcs().items()}
        q_recv_count = 0

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
                name, raw = dat
                worker_funcs[name](*raw, serialized=True)
                q_recv_count += 1
                share_data.q_recv_count = q_recv_count

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


class _TrainerInterruptThread(RunCallback):
    def __init__(self, memory_ps: threading.Thread, parameter_ps: threading.Thread, share_data: _ShareData) -> None:
        self.memory_ps = memory_ps
        self.parameter_ps = parameter_ps
        self.share_data = share_data

    def on_train_before(self, context: RunContext, state: RunStateTrainer, **kwargs) -> bool:
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
class _TrainerInterruptNoThread(RunCallback):
    def __init__(
        self,
        manager: ServerManager,
        trainer_parameter_send_interval: int,
        memory: RLMemory,
    ) -> None:
        self.memory_receiver = manager.get_memory_receiver()
        self.parameter_writer = manager.get_parameter_writer()
        self.task_manager = manager.get_task_manager()
        self.trainer_parameter_send_interval = trainer_parameter_send_interval
        self.q_recv_count = 0

        self.keepalive_t0 = 0
        self.sync_parameter_t0 = time.time()
        self.start_train_count = self.task_manager.get_train_count()

        self.worker_funcs = {k: v[0] for k, v in memory.get_worker_funcs().items()}

    def on_train_after(self, context: RunContext, state: RunStateTrainer, **kwargs) -> bool:
        state.memory = cast(RLMemory, state.memory)

        # --- recv memory -> add
        if not self.memory_receiver.is_connected:
            self.memory_receiver.ping()
        if self.memory_receiver.is_connected:
            dat = self.memory_receiver.memory_recv()
            if dat is not None:
                name, raw = dat
                self.worker_funcs[name](*raw, serialized=True)
                self.q_recv_count += 1
                state.trainer_recv_q = self.q_recv_count
        else:
            dat = None

        if state.is_step_trained:
            # --- sync parameter
            if time.time() - self.sync_parameter_t0 > self.trainer_parameter_send_interval:
                self.sync_parameter_t0 = time.time()

                params = state.parameter.backup(to_cpu=True)
                if params is not None:
                    self.parameter_writer.parameter_update(params)
                    state.sync_trainer += 1
        else:
            # 学習してなく、dataの受信もなければ待つ
            if dat is None:
                time.sleep(1)

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

    def on_trainer_end(self, context: RunContext, state: RunStateTrainer, **kwargs):
        self.task_manager.set_trainer("q_recv_count", str(self.q_recv_count))
        _keepalive(
            self.task_manager,
            self.start_train_count,
            state.trainer.get_train_count(),
        )


# ------------------------------------------
# run
# ------------------------------------------
def _run_trainer(manager: ServerManager, task_config: TaskConfig, parameter: RLParameter):
    task_manager = manager.get_task_manager()
    parameter_writer = manager.get_parameter_writer()
    callbacks = task_config.callbacks[:]

    # --- parameter
    params = parameter.backup(to_cpu=True)
    if params is not None:
        parameter_writer.parameter_update(params)

    # --- memory
    memory = task_config.context.rl_config.make_memory()

    memory_ps = None
    parameter_ps = None
    share_data = None
    try:
        # --- thread
        if task_config.enable_trainer_thread:
            share_data = _ShareData()

            _manager_copy_args = manager._copy_args()
            memory_ps = threading.Thread(
                target=_memory_communicate,
                args=(_manager_copy_args, memory, share_data),
            )
            parameter_ps = threading.Thread(
                target=_parameter_communicate,
                args=(
                    _manager_copy_args,
                    parameter,
                    share_data,
                    task_config.trainer_parameter_send_interval,
                ),
            )
            memory_ps.start()
            parameter_ps.start()
            callbacks.append(_TrainerInterruptThread(memory_ps, parameter_ps, share_data))
        else:
            callbacks.append(
                _TrainerInterruptNoThread(
                    manager,
                    task_config.trainer_parameter_send_interval,
                    memory,
                )
            )

        # --- play
        context = task_config.context
        context.training = True
        context.distributed = True
        context.train_only = False
        trainer = context.rl_config.make_trainer(parameter, memory)
        core_train_only.play_trainer_only(context, trainer, callbacks)

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


def _task_assign(task_manager: TaskManager):
    if task_manager.get_status() != "ACTIVE":
        return None

    # --- init parameter & check
    task_config = task_manager.get_config()
    if task_config is None:
        return None
    task_config.context.run_name = RunNameTypes.trainer
    parameter = task_config.context.rl_config.make_parameter()
    task_manager.read_parameter(parameter)

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
    return task_config, parameter


def run_forever(
    redis_params: RedisParameters,
    memory_params: Optional[IMemoryServerParameters] = None,
    keepalive_interval: int = 10,
    keepalive_threshold: int = 101,
    callbacks: List[TrainerServerCallback] = [],
    framework: str = "tensorflow",
    device: str = "AUTO",
    set_CUDA_VISIBLE_DEVICES_if_CPU: bool = True,
    tf_enable_memory_growth: bool = True,
    tf_mixed_precision_policy_name: str = "",
    run_once: bool = False,
    is_remote_memory_purge: bool = True,
):
    used_device_tf, used_device_torch = setup_device(
        framework,
        device,
        set_CUDA_VISIBLE_DEVICES_if_CPU,
        tf_enable_memory_growth,
        tf_mixed_precision_policy_name,
    )
    task_manager_params = TaskManagerParams(
        "trainer",
        keepalive_interval,
        keepalive_threshold,
        framework=framework,
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
            run_params = _task_assign(task_manager)
            if run_params is not None:
                print(f"train start: {uid}")
                logger.info(f"train start: {uid}")
                task_manager.setup_memory(memory_receiver, is_remote_memory_purge)
                _run_trainer(manager, *run_params)
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
