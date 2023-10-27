import logging
import queue
import threading
import time
import traceback
from typing import Any, List, Optional, cast

import srl
from srl.base.rl.base import IRLMemoryTrainer, RLMemory, RLParameter, RLTrainer
from srl.base.run.context import RunNameTypes
from srl.runner.callback import TrainerCallback
from srl.runner.distribution.manager import DistributedManager
from srl.runner.runner import Runner, TaskConfig

logger = logging.getLogger(__name__)


class _RLTrainer(RLTrainer):
    def __init__(self, base_trainer: RLTrainer):
        super().__init__(base_trainer.config, base_trainer.parameter, base_trainer.memory)
        self.base_trainer = base_trainer

        self.sync_count = 0
        self.count_for_sync = 0

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            time.sleep(1)
            return
        memory_sample_return = self.memory.sample(self.base_trainer.batch_size, self.train_count)
        _prev_train_count = self.base_trainer.train_count
        self.base_trainer.train_on_batchs(memory_sample_return)
        self.count_for_sync += self.base_trainer.train_count - _prev_train_count

        # --- infos
        self.train_count = self.base_trainer.train_count
        self.train_info = self.base_trainer.train_info
        self.train_info["mp_sync"] = self.sync_count

    def train_on_batchs(self, memory_sample_return) -> None:
        raise NotImplementedError("not used")


class _TrainerRLMemory(IRLMemoryTrainer):
    def __init__(self, base_memory: RLMemory):
        self.base_memory = base_memory
        self.q_batch = queue.Queue()
        self.q_update = queue.Queue()
        self.trainer: RLTrainer

    def add(self, *args) -> None:
        self.base_memory.add(*args)

    def loop_update(self):
        if not self.base_memory.is_warmup_needed():
            if self.q_batch.qsize() < 5:
                self.q_batch.put(self.base_memory.sample(self.trainer.batch_size, self.trainer.train_count))
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


def _server_communicate(
    manager_args,
    task_id: str,
    memory: _TrainerRLMemory,
    parameter: RLParameter,
    trainer: _RLTrainer,
    trainer_parameter_send_interval_by_train_count: int,
    enable_prepare_batch: bool,
):
    try:
        manager = DistributedManager.create(*manager_args)
        while True:
            # --- add memory
            dat = manager.memory_recv(task_id)
            if dat is not None:
                memory.add(*dat)
            if enable_prepare_batch:
                memory.loop_update()

            # --- sync parameter
            if trainer.count_for_sync > trainer_parameter_send_interval_by_train_count:
                trainer.count_for_sync = 0

                params = parameter.backup()
                if params is not None:
                    manager.parameter_update(task_id, params)
                    trainer.sync_count += 1

            # --- keepalive
            if manager.keepalive(task_id):
                if manager.task_get_status(task_id) == "END":
                    break

                manager.task_set_trainer(task_id, "train", trainer.get_train_count())
                manager.task_set_trainer(task_id, "memory", memory.length())

    except Exception:
        logger.info(traceback.format_exc())
    logger.info("trainer thread end.")


class _TrainerInterrupt(TrainerCallback):
    def __init__(self, server_ps: threading.Thread) -> None:
        self.server_ps = server_ps

    def on_trainer_train_end(self, runner: Runner) -> bool:
        return not self.server_ps.is_alive()


def _run_trainer(manager: DistributedManager, task_id: str, task_config: TaskConfig):
    task_config.context.run_name = RunNameTypes.trainer

    # --- memory
    memory = srl.make_memory(task_config.context.rl_config)
    if task_config.context.enable_prepare_batch:
        memory = cast(RLMemory, _TrainerRLMemory(memory))

    # --- runner
    runner = srl.Runner(
        task_config.context.env_config,
        task_config.context.rl_config,
        task_config.config,
        task_config.context,
        memory=memory,
    )

    # --- parameter
    parameter = runner.make_parameter(is_load=False)
    params = manager.parameter_read(task_id)
    if params is None:
        logger.warning("Missing initial parameters")
    else:
        parameter.restore(params)

    try:
        trainer = _RLTrainer(runner.make_trainer())
        if task_config.context.enable_prepare_batch:
            cast(_TrainerRLMemory, memory).trainer = trainer

        # --- connect
        server_ps = threading.Thread(
            target=_server_communicate,
            args=(
                manager.create_args(),
                task_id,
                memory,
                parameter,
                trainer,
                task_config.context.trainer_parameter_send_interval_by_train_count,
                task_config.context.enable_prepare_batch,
            ),
        )
        server_ps.start()

        # --- play
        runner.context.callbacks.append(_TrainerInterrupt(server_ps))
        runner.core_play(trainer_only=True, trainer=trainer)

    finally:
        # --- last params
        params = parameter.backup()
        assert params is not None
        manager.parameter_update(task_id, params)

        # --- 終了はtrainerで
        manager.task_end(task_id)


class TrainerServerCallback:
    def on_polling(self) -> Optional[bool]:
        """If return is True, it will end intermediate stop."""
        return False


def run_forever(
    host: str,
    port: int = 6379,
    redis_kwargs: dict = {},
    keepalive_interval: int = 10,
    callbacks: List[TrainerServerCallback] = [],
):
    manager = DistributedManager(host, port, redis_kwargs, keepalive_interval)
    manager.server_ping()
    manager.set_user("trainer")

    print(f"wait trainer: {manager.uid}")
    while True:
        try:
            time.sleep(1)

            # --- callback
            _stop_flags = [c.on_polling() for c in callbacks]
            if True in _stop_flags:
                break

            # --- board check
            task_id, task_config, _ = manager.task_assign_by_my_id()

            # --- train loop
            if task_config is not None:
                try:
                    print(f"train start: {manager.uid}")
                    logger.info(f"train start: {manager.uid}")
                    _run_trainer(manager, task_id, task_config)
                    logger.info(f"train end: {manager.uid}")
                finally:
                    print(f"wait trainer: {manager.uid}")

        except Exception:
            logger.error(traceback.format_exc())
