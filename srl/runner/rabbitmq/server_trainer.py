import logging
import queue
import threading
import time
import traceback
from typing import Any, List, cast

from pika.adapters.blocking_connection import BlockingChannel

import srl
from srl.base.rl.base import IRLMemoryTrainer, RLMemory, RLParameter, RLTrainer
from srl.base.run.context import RunNameTypes
from srl.runner.callback import TrainerCallback
from srl.runner.rabbitmq.rabbitmq_manager import RabbitMQManager
from srl.runner.runner import RunnerMPData

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


def _mq_thread(
    mq_args,
    client_id,
    memory: _TrainerRLMemory,
    parameter: RLParameter,
    trainer: _RLTrainer,
    trainer_parameter_send_interval_by_train_count: int,
    enable_prepare_batch: bool,
):
    mq = RabbitMQManager.create(*mq_args)
    try:
        mq.create_queue_once_if_not_exists(f"memory_{client_id}")
        mq.purge_queue_once(f"memory_{client_id}")

        while True:
            # --- add memory
            mq.create_queue_once_if_not_exists(f"memory_{client_id}")
            body = mq.recv_once(f"memory_{client_id}")
            if body is not None:
                memory.add(*body)
            if enable_prepare_batch:
                memory.loop_update()

            # --- sync parameter
            if trainer.count_for_sync > trainer_parameter_send_interval_by_train_count:
                trainer.count_for_sync = 0

                params = parameter.backup()
                if params is not None:
                    if mq.send_fanout_once("parameter", params):
                        trainer.sync_count += 1

            # --- keepalive
            if mq.keepalive():
                if not mq.taskcheck_alive_task(client_id):
                    break

    except Exception:
        logger.info(traceback.format_exc())
        logger.info("trainer thread end.")


class _TrainerInterrupt(TrainerCallback):
    def __init__(self, mq_ps: threading.Thread) -> None:
        self.mq_ps = mq_ps

    def intermediate_stop(self, runner) -> bool:
        return not self.mq_ps.is_alive()


def _run_trainer(mq: RabbitMQManager, mp_data: RunnerMPData, client_id):
    mp_data.context.run_name = RunNameTypes.trainer

    # --- memory
    memory = srl.make_memory(mp_data.context.rl_config)
    if mp_data.context.enable_prepare_batch:
        memory = cast(RLMemory, _TrainerRLMemory(memory))

    # --- runner
    runner = srl.Runner(
        mp_data.context.env_config, mp_data.context.rl_config, mp_data.config, mp_data.context, memory=memory
    )
    parameter = runner.make_parameter(is_load=False)

    try:
        trainer = _RLTrainer(runner.make_trainer())
        if mp_data.context.enable_prepare_batch:
            memory.trainer = trainer  # type: ignore

        # --- connect
        mq_ps = threading.Thread(
            target=_mq_thread,
            args=(
                mq.copy_args(),
                client_id,
                memory,
                parameter,
                trainer,
                mp_data.context.trainer_parameter_send_interval_by_train_count,
                mp_data.context.enable_prepare_batch,
            ),
        )
        mq_ps.start()

        # --- play
        runner.context.callbacks.append(_TrainerInterrupt(mq_ps))
        runner.core_play(trainer_only=True, trainer=trainer)

    finally:
        # --- last params
        try:
            params = parameter.backup()
            if params is not None:
                mq.create_queue_once_if_not_exists(f"last_parameter_{client_id}")
                mq.send_loop(f"last_parameter_{client_id}", params)
        except Exception:
            logger.warning(traceback.format_exc())


class TrainerServerCallback:
    def on_polling(self, channel: BlockingChannel) -> None:
        pass  # do nothing

    # 外部から途中停止用
    def intermediate_stop(self, channel: BlockingChannel) -> bool:
        return False


def run_forever(
    host: str,
    port: int = 5672,
    username: str = "guest",
    password: str = "guest",
    virtual_host: str = "/",
    callbacks: List[TrainerServerCallback] = [],
):
    mq = RabbitMQManager(host, port, username, password, virtual_host)
    mq.join("trainer")
    mq.board_update({"status": "", "client": ""})

    print(f"wait trainer: {mq.uid}")
    while True:
        time.sleep(1)
        if mq.keepalive():
            mq.health_check()

        try:
            # --- callback
            # [c.on_polling(mq) for c in callbacks]
            # for c in callbacks:
            #    if c.intermediate_stop(mq):
            #        break

            # --- board check
            client_id, mp_data, _ = mq.taskcheck_if_my_id_assigned()

            # --- train loop
            if client_id != "":
                assert mp_data is not None
                logger.info("trainer start")
                _run_trainer(mq, mp_data, client_id)
                logger.info("trainer end")
                print(f"wait trainer: {mq.uid}")

            # --- boardを更新
            mq.board_update({"status": "", "client": ""})

        except Exception:
            logger.error(traceback.format_exc())
