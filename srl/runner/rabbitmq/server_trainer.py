import logging
import pickle
import threading
import time
import traceback

import pika
from pika.adapters.blocking_connection import BlockingChannel

import srl
from srl.base.rl.base import RLMemory, RLParameter, RLTrainer
from srl.base.run.data import RunNameTypes
from srl.runner.callback import TrainerCallback
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


def _connect_mq(
    pika_params,
    memory: RLMemory,
    parameter: RLParameter,
    trainer: _RLTrainer,
    trainer_parameter_send_interval_by_train_count: int,
):
    try:
        with pika.BlockingConnection(pika_params) as connection:
            channel = connection.channel()

            channel.queue_declare(queue="memory")
            channel.queue_purge(queue="memory")

            while True:
                # --- add memory
                method_frame, header_frame, body = channel.basic_get(queue="memory", auto_ack=True)
                if body is not None:
                    args = pickle.loads(body)
                    memory.add(*args)

                # --- sync parameter
                if trainer.count_for_sync > trainer_parameter_send_interval_by_train_count:
                    trainer.count_for_sync = 0

                    params = parameter.backup()
                    if params is not None:
                        channel.basic_publish(exchange="parameter", routing_key="", body=pickle.dumps(params))
                        trainer.sync_count += 1

                # --- alive
                method_frame, header_frame, body = channel.basic_get(queue="trainer_end", auto_ack=True)
                if body is not None:
                    break

    except Exception:
        logger.info(traceback.format_exc())
        logger.info("trainer thread end.")


class _TrainerInterrupt(TrainerCallback):
    def __init__(self, mq_ps: threading.Thread) -> None:
        self.mq_ps = mq_ps

    def intermediate_stop(self, runner) -> bool:
        return not self.mq_ps.is_alive()


def _run_trainer(channel: BlockingChannel, mp_data: RunnerMPData, pika_params):
    mp_data.context.run_name = RunNameTypes.main

    # --- runner
    runner = srl.Runner(mp_data.env_config, mp_data.rl_config, mp_data.config, mp_data.context)
    parameter = runner.make_parameter(is_load=False)
    memory = runner.make_memory()
    try:
        trainer = _RLTrainer(runner.make_trainer())

        # --- connect
        mq_ps = threading.Thread(
            target=_connect_mq,
            args=(
                pika_params,
                memory,
                parameter,
                trainer,
                mp_data.context.trainer_parameter_send_interval_by_train_count,
            ),
        )
        mq_ps.start()

        # --- play
        runner.context.callbacks.append(_TrainerInterrupt(mq_ps))
        runner.core_play(trainer_only=True, trainer=trainer)

    finally:
        with pika.BlockingConnection(pika_params) as connection:
            channel = connection.channel()
            channel.basic_publish(exchange="end", routing_key="", body="STOP")

            # --- send last params
            body = pickle.dumps(parameter.backup())
            channel.basic_publish(exchange="", routing_key="last_parameter", body=body)


def run_forever(
    host: str,
    port: int = 5672,
    user: str = "guest",
    password: str = "guest",
):
    # thread配下でも共有してるっぽい、個別に接続する
    pika_params = pika.ConnectionParameters(host, port, credentials=pika.PlainCredentials(user, password))
    with pika.BlockingConnection(pika_params) as connection:
        channel = connection.channel()

        # start
        channel.exchange_declare(exchange="start", exchange_type="fanout")
        channel.queue_declare(queue="trainer_start")
        channel.queue_bind(exchange="start", queue="trainer_start")

        # end
        channel.exchange_declare(exchange="end", exchange_type="fanout")
        channel.queue_declare(queue="trainer_end")
        channel.queue_bind(exchange="end", queue="trainer_end")

        # parameter
        channel.exchange_declare(exchange="parameter", exchange_type="fanout")

        # queue
        channel.queue_declare(queue="last_parameter")

    while True:
        # --- wait start
        with pika.BlockingConnection(pika_params) as connection:
            channel = connection.channel()
            channel.queue_purge(queue="trainer_start")
            print("wait trainer")
            while True:
                time.sleep(1)
                method_frame, header_frame, body = channel.basic_get(queue="trainer_start", auto_ack=True)
                if body is None:
                    continue
                body = pickle.loads(body)
                channel.queue_purge(queue="trainer_end")
                channel.queue_purge(queue="last_parameter")
                break

        # --- start train
        try:
            logger.info("trainer start")
            _run_trainer(channel, body, pika_params)
        except Exception:
            logger.error(traceback.format_exc())
        logger.info("trainer end")
