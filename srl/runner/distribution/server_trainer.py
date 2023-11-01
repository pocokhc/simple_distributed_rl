import logging
import queue
import threading
import time
import traceback
from typing import Any, List, cast

import srl
from srl.base.rl.base import IRLMemoryTrainer, RLMemory, RLParameter
from srl.base.run.context import RunNameTypes
from srl.runner.callback import TrainerCallback
from srl.runner.distribution.callback import TrainerServerCallback
from srl.runner.distribution.manager import DistributedManager, ServerParameters
from srl.runner.runner import Runner, TaskConfig

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
class _TrainerRLMemoryThreadPrepareBatch(IRLMemoryTrainer):
    def __init__(self, base_memory: RLMemory, batch_size: int, share_dict: dict):
        # recv,warmup両方threadなので、両方待つ場合は待機
        self.base_memory = base_memory
        self.batch_size = batch_size
        self.share_dict = share_dict
        self.q_batch = queue.Queue()
        self.q_update = queue.Queue()

    def recv(self, dat) -> None:
        if dat is not None:
            self.base_memory.add(*dat)
        if dat is None and self.base_memory.is_warmup_needed():
            time.sleep(1)
        if not self.base_memory.is_warmup_needed():
            if self.q_batch.qsize() < 5:
                self.q_batch.put(self.base_memory.sample(self.batch_size, self.share_dict["train_count"]))
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


class _TrainerRLMemoryThread(IRLMemoryTrainer):
    def __init__(self, base_memory: RLMemory):
        # thread(recv) は受信できなければ待機
        # main(is_warmup_needed) はwarmup中なら待機
        self.base_memory = base_memory

    def recv(self, dat) -> None:
        if dat is None:
            time.sleep(1)
        else:
            self.base_memory.add(*dat)

    def length(self) -> int:
        return self.base_memory.length()

    def is_warmup_needed(self) -> bool:
        if self.base_memory.is_warmup_needed():
            time.sleep(1)
            return True
        return False

    def sample(self, batch_size: int, step: int) -> Any:
        return self.base_memory.sample(batch_size, step)

    def update(self, memory_update_args: Any) -> None:
        self.base_memory.update(memory_update_args)


def _memory_communicate(
    manager_args,
    task_id: str,
    memory: _TrainerRLMemoryThread,
    share_dict: dict,
    th_exit_event: threading.Event,
):
    try:
        manager = DistributedManager.create(*manager_args)
        remote_memory = manager.create_memory_connector()

        while not th_exit_event.is_set():
            dat = remote_memory.memory_recv(task_id)
            memory.recv(dat)

    except Exception:
        share_dict["th_error"] = traceback.format_exc()
    finally:
        share_dict["end_signal"] = True
        logger.info("trainer memory thread end.")


def _parameter_communicate(
    manager_args,
    task_id: str,
    memory: _TrainerRLMemoryThread,
    parameter: RLParameter,
    share_dict: dict,
    trainer_parameter_send_interval: int,
    th_exit_event: threading.Event,
):
    try:
        manager = DistributedManager.create(*manager_args)

        t0 = time.time()
        while not th_exit_event.is_set():
            # --- sync parameter
            if time.time() - t0 > trainer_parameter_send_interval:
                t0 = time.time()

                params = parameter.backup()
                if params is not None:
                    manager.parameter_update(task_id, params)
                    share_dict["sync_count"] += 1

            # --- keepalive
            if manager.keepalive(task_id):
                manager.task_set_trainer(task_id, "train", str(share_dict["train_count"]))
                manager.task_set_trainer(task_id, "memory", str(memory.length()))
                if manager.task_is_dead(task_id):
                    logger.info(f"task is dead: {task_id}")
                    break

            time.sleep(1)

        manager.keepalive(task_id, do_now=True)
        manager.task_set_trainer(task_id, "train", str(share_dict["train_count"]))
        manager.task_set_trainer(task_id, "memory", str(memory.length()))

    except Exception:
        share_dict["th_error"] = traceback.format_exc()
    finally:
        share_dict["end_signal"] = True
        logger.info("trainer parameter thread end.")


class _TrainerInterruptThread(TrainerCallback):
    def __init__(self, memory_ps: threading.Thread, parameter_ps: threading.Thread, share_dict: dict) -> None:
        self.memory_ps = memory_ps
        self.parameter_ps = parameter_ps
        self.share_dict = share_dict

    def on_trainer_loop(self, runner: Runner) -> bool:
        assert runner.state.trainer is not None
        self.share_dict["train_count"] = runner.state.trainer.get_train_count()
        runner.state.sync_trainer = self.share_dict["sync_count"]
        if not self.memory_ps.is_alive():
            self.share_dict["end_signal"] = True
        if not self.parameter_ps.is_alive():
            self.share_dict["end_signal"] = True
        return self.share_dict["end_signal"]


# ------------------------------------------
# no thread(add -> sample -> train -> update)
# ------------------------------------------
class _TrainerInterruptManager(TrainerCallback):
    def __init__(
        self,
        manager: DistributedManager,
        task_id,
        memory: RLMemory,
        trainer_parameter_send_interval: int,
    ) -> None:
        self.manager = manager
        self.remote_memory = self.manager.create_memory_connector()
        self.task_id = task_id
        self.memory = memory
        self.trainer_parameter_send_interval = trainer_parameter_send_interval
        self.t0 = time.time()

    def on_trainer_loop(self, runner: Runner) -> bool:
        # --- recv memory
        dat = self.remote_memory.memory_recv(self.task_id)
        if dat is not None:
            assert runner.state.memory is not None
            self.memory.add(*dat)

        # no warmupとmemory emptyなら待つ
        if dat is None and self.memory.is_warmup_needed():
            time.sleep(1)

        # --- sync parameter
        if runner.state.is_step_trained:
            if time.time() - self.t0 > self.trainer_parameter_send_interval:
                self.t0 = time.time()

                assert runner.state.parameter is not None
                params = runner.state.parameter.backup()
                if params is not None:
                    self.manager.parameter_update(self.task_id, params)
                    runner.state.sync_trainer += 1

        # --- keepalive
        if self.manager.keepalive(self.task_id):
            assert runner.state.trainer is not None
            self.manager.task_set_trainer(self.task_id, "train", str(runner.state.trainer.get_train_count()))
            self.manager.task_set_trainer(self.task_id, "memory", str(self.memory.length()))
            if self.manager.task_is_dead(self.task_id):
                logger.info(f"task is dead: {self.task_id}")
                return True
        return False

    def on_trainer_end(self, runner: Runner):
        assert runner.state.trainer is not None
        self.manager.keepalive(self.task_id, do_now=True)
        self.manager.task_set_trainer(self.task_id, "train", str(runner.state.trainer.get_train_count()))
        self.manager.task_set_trainer(self.task_id, "memory", str(self.memory.length()))


# --------------------------------
# main
# --------------------------------
def _run_trainer(manager: DistributedManager, task_id: str, task_config: TaskConfig):
    task_config.context.run_name = RunNameTypes.trainer

    # --- runner
    runner = srl.Runner(
        task_config.context.env_config,
        task_config.context.rl_config,
        task_config.config,
        task_config.context,
    )

    # --- parameter
    parameter = runner.make_parameter(is_load=False)
    params = manager.parameter_read(task_id)
    if params is None:
        logger.warning("Missing initial parameters")
    else:
        parameter.restore(params)

    # --- memory
    memory = runner.make_memory(is_load=False)

    memory_ps = None
    parameter_ps = None
    th_exit_event = None
    share_dict = {}
    try:
        # --- thread
        if task_config.config.dist_enable_trainer_thread:
            share_dict = {
                "sync_count": 0,
                "train_count": 0,
                "end_signal": False,
                "th_error": "",
            }
            if task_config.config.dist_enable_prepare_sample_batch:
                batch_size = getattr(task_config.context.rl_config, "batch_size", 1)
                memory = _TrainerRLMemoryThreadPrepareBatch(memory, batch_size, share_dict)
            else:
                memory = _TrainerRLMemoryThread(memory)

            th_exit_event = threading.Event()
            memory_ps = threading.Thread(
                target=_memory_communicate,
                args=(
                    manager.create_args(),
                    task_id,
                    memory,
                    share_dict,
                    th_exit_event,
                ),
            )
            parameter_ps = threading.Thread(
                target=_parameter_communicate,
                args=(
                    manager.create_args(),
                    task_id,
                    memory,
                    parameter,
                    share_dict,
                    task_config.config.trainer_parameter_send_interval,
                    th_exit_event,
                ),
            )
            memory_ps.start()
            parameter_ps.start()
            runner.context.callbacks.append(_TrainerInterruptThread(memory_ps, parameter_ps, share_dict))
        else:
            runner.context.callbacks.append(
                _TrainerInterruptManager(
                    manager,
                    task_id,
                    memory,
                    task_config.config.trainer_parameter_send_interval,
                )
            )

        # --- play
        runner.core_play(
            trainer_only=True,
            parameter=parameter,
            memory=cast(RLMemory, memory),
            trainer=None,
            workers=None,
        )
    except Exception:
        raise
    finally:
        # --- last params
        params = parameter.backup()
        if params is not None:
            manager.parameter_update(task_id, params)

        # --- 終了はtrainerで
        manager.task_end(task_id)

        if th_exit_event is not None:
            th_exit_event.set()
            assert memory_ps is not None
            assert parameter_ps is not None
            memory_ps.join(timeout=10)
            parameter_ps.join(timeout=10)
            if share_dict["th_error"] != "":
                raise ValueError(share_dict["th_error"])


def run_forever(
    parameter: ServerParameters,
    callbacks: List[TrainerServerCallback] = [],
    framework: str = "tensorflow",
    device: str = "AUTO",
    run_once: bool = False,
):
    used_device_tf, used_device_torch = Runner.setup_device(framework, device)

    manager = DistributedManager(parameter)
    assert manager.ping()
    assert manager.create_memory_connector().ping()
    manager.set_user("trainer")

    print(f"wait trainer: {manager.uid}")
    while True:
        try:
            time.sleep(1)

            # --- callback
            _stop_flags = [c.on_polling() for c in callbacks]
            if True in _stop_flags:
                break

            # --- task check
            task_id, _ = manager.task_assign_by_my_id()
            if task_id != "":
                try:
                    print(f"train start: {manager.uid}")
                    logger.info(f"train start: {manager.uid}")
                    task_config = manager.task_get_config(task_id)
                    assert task_config is not None
                    task_config.context.create_controller().set_device(used_device_tf, used_device_torch)
                    task_config.context.used_device_tf = used_device_tf
                    task_config.context.used_device_torch = used_device_torch
                    _run_trainer(manager, task_id, task_config)
                    logger.info(f"train end: {manager.uid}")
                    if run_once:
                        break
                except Exception:
                    raise
                finally:
                    print(f"wait trainer: {manager.uid}")

        except Exception:
            if run_once:
                raise
            else:
                logger.error(traceback.format_exc())
