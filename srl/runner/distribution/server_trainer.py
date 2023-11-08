import logging
import queue
import threading
import time
import traceback
from typing import Any, List, Optional, cast

import srl
from srl.base.rl.base import IRLMemoryTrainer, RLMemory, RLParameter
from srl.base.run.context import RunNameTypes
from srl.runner.callback import TrainerCallback
from srl.runner.distribution.callback import TrainerServerCallback
from srl.runner.distribution.connectors.imemory import IServerParameters
from srl.runner.distribution.connectors.parameters import RedisParameters
from srl.runner.distribution.manager import DistributedManager
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
            time.sleep(0.1)
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


def _memory_communicate(
    manager_args,
    memory: RLMemory,
    share_dict: dict,
    enable_prepare_sample_batch,
):
    try:
        manager = DistributedManager.create(*manager_args)
        remote_memory = manager.create_memory_connector()

        q_recv_count = 0

        if enable_prepare_sample_batch:
            memory_th = cast(_TrainerRLMemoryThreadPrepareBatch, memory)
            while not share_dict["end_signal"]:
                dat = remote_memory.memory_recv()
                memory_th.recv(dat)
                if dat is not None:
                    q_recv_count += 1
                    share_dict["q_recv_count"] = q_recv_count
        else:
            while not share_dict["end_signal"]:
                dat = remote_memory.memory_recv()
                if dat is None:
                    time.sleep(0.1)
                else:
                    q_recv_count += 1
                    share_dict["q_recv_count"] = q_recv_count
                    memory.add(*dat)

    except Exception:
        share_dict["th_error"] = traceback.format_exc()
    finally:
        share_dict["end_signal"] = True
        logger.info("trainer memory thread end.")


def _parameter_communicate(
    manager_args,
    memory: RLMemory,
    parameter: RLParameter,
    share_dict: dict,
    trainer_parameter_send_interval: int,
):
    try:
        manager = DistributedManager.create(*manager_args)

        while not share_dict["end_signal"]:
            # --- sync parameter
            time.sleep(trainer_parameter_send_interval)
            params = parameter.backup(to_cpu=True)
            if params is not None:
                manager.parameter_update(params)
                share_dict["sync_count"] += 1

            # --- keepalive
            manager.task_set_trainer("q_recv_count", str(share_dict["q_recv_count"]))

            if manager.keepalive():
                manager.task_set_trainer("train", str(share_dict["train_count"]))
                manager.task_set_trainer("memory", str(memory.length()))
                if manager.task_is_dead():
                    logger.info("task is dead")
                    break

        manager.keepalive(do_now=True)
        manager.task_set_trainer("q_recv_count", str(share_dict["q_recv_count"]))
        manager.task_set_trainer("train", str(share_dict["train_count"]))
        manager.task_set_trainer("memory", str(memory.length()))

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
        if not runner.state.is_step_trained:
            # warmupなら待機
            time.sleep(1)

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
        trainer_parameter_send_interval: int,
    ) -> None:
        self.manager = manager
        self.remote_memory = self.manager.create_memory_connector()
        self.trainer_parameter_send_interval = trainer_parameter_send_interval
        self.q_recv_count = 0
        self.t0 = time.time()

    def on_trainer_loop(self, runner: Runner) -> bool:
        assert runner.state.memory is not None
        runner.state.memory = cast(RLMemory, runner.state.memory)

        # --- recv memory
        dat = self.remote_memory.memory_recv()
        if dat is not None:
            self.q_recv_count += 1
            runner.state.memory.add(*dat)

        # no warmupとmemory emptyなら待つ
        if dat is None and runner.state.memory.is_warmup_needed():
            time.sleep(1)

        # --- sync parameter
        if runner.state.is_step_trained:
            if time.time() - self.t0 > self.trainer_parameter_send_interval:
                self.t0 = time.time()

                assert runner.state.parameter is not None
                params = runner.state.parameter.backup(to_cpu=True)
                if params is not None:
                    self.manager.parameter_update(params)
                    runner.state.sync_trainer += 1

        # --- keepalive
        if self.manager.keepalive():
            assert runner.state.trainer is not None
            assert runner.state.memory is not None
            self.manager.task_set_trainer("train", str(runner.state.trainer.get_train_count()))
            self.manager.task_set_trainer("memory", str(runner.state.memory.length()))
            self.manager.task_set_trainer("q_recv_count", str(self.q_recv_count))
            if self.manager.task_is_dead():
                logger.info("task is dead")
                return True
        return False

    def on_trainer_end(self, runner: Runner):
        assert runner.state.trainer is not None
        assert runner.state.memory is not None
        self.manager.keepalive(do_now=True)
        self.manager.task_set_trainer("train", str(runner.state.trainer.get_train_count()))
        self.manager.task_set_trainer("memory", str(runner.state.memory.length()))
        self.manager.task_set_trainer("q_recv_count", str(self.q_recv_count))


# --------------------------------
# main
# --------------------------------
def _run_trainer(manager: DistributedManager, task_config: TaskConfig):
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
    params = manager.parameter_read()
    if params is None:
        logger.warning("Missing initial parameters")
    else:
        parameter.restore(params, from_cpu=True)

    # --- memory
    memory = runner.make_memory(is_load=False)

    memory_ps = None
    parameter_ps = None
    share_dict = {}
    try:
        # --- thread
        if task_config.config.dist_enable_trainer_thread:
            share_dict = {
                "sync_count": 0,
                "train_count": 0,
                "q_recv_count": 0,
                "end_signal": False,
                "th_error": "",
            }
            if task_config.config.dist_enable_prepare_sample_batch:
                batch_size = getattr(task_config.context.rl_config, "batch_size", 1)
                memory = _TrainerRLMemoryThreadPrepareBatch(memory, batch_size, share_dict)

            memory_ps = threading.Thread(
                target=_memory_communicate,
                args=(
                    manager.create_args(),
                    memory,
                    share_dict,
                    task_config.config.dist_enable_prepare_sample_batch,
                ),
            )
            parameter_ps = threading.Thread(
                target=_parameter_communicate,
                args=(
                    manager.create_args(),
                    memory,
                    parameter,
                    share_dict,
                    task_config.config.trainer_parameter_send_interval,
                ),
            )
            memory_ps.start()
            parameter_ps.start()
            runner.context.callbacks.append(_TrainerInterruptThread(memory_ps, parameter_ps, share_dict))
        else:
            runner.context.callbacks.append(
                _TrainerInterruptManager(
                    manager,
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
        params = parameter.backup(to_cpu=True)
        if params is not None:
            manager.parameter_update(params)

        # --- 終了はtrainerで
        manager.task_end()

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
    callbacks: List[TrainerServerCallback] = [],
    framework: str = "tensorflow",
    device: str = "AUTO",
    run_once: bool = False,
    is_remote_memory_purge: bool = True,
):
    used_device_tf, used_device_torch = Runner.setup_device(framework, device)

    manager = DistributedManager(redis_parameter, memory_parameter)
    assert manager.ping()
    manager.set_user("trainer")

    print(f"wait trainer: {manager.uid}")
    logger.info(f"wait trainer: {manager.uid}")
    while True:
        try:
            time.sleep(1)

            # --- callback
            _stop_flags = [c.on_polling() for c in callbacks]
            if True in _stop_flags:
                break

            # --- task check
            is_assigned, _ = manager.task_assign_by_my_id()
            if is_assigned:
                print(f"train start: {manager.uid}")
                logger.info(f"train start: {manager.uid}")
                task_config = manager.task_get_config()
                assert task_config is not None
                task_config.context.create_controller().set_device(used_device_tf, used_device_torch)
                task_config.context.used_device_tf = used_device_tf
                task_config.context.used_device_torch = used_device_torch
                if is_remote_memory_purge:
                    manager.create_memory_connector().memory_purge()
                _run_trainer(manager, task_config)
                logger.info(f"train end: {manager.uid}")
                if run_once:
                    break
                print(f"wait trainer: {manager.uid}")
                logger.info(f"wait trainer: {manager.uid}")

        except Exception:
            if run_once:
                raise
            else:
                logger.error(traceback.format_exc())
                print(f"wait trainer: {manager.uid}")
                logger.info(f"wait trainer: {manager.uid}")
