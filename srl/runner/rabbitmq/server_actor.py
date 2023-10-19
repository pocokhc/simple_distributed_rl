import logging
import time
import traceback
from typing import List, cast

from pika.adapters.blocking_connection import BlockingChannel

import srl
from srl.base.rl.base import IRLMemoryWorker, RLMemory, RLParameter
from srl.base.run.context import RunNameTypes
from srl.runner.callback import Callback
from srl.runner.rabbitmq.rabbitmq_manager import RabbitMQManager
from srl.runner.runner import RunnerMPData

logger = logging.getLogger(__name__)


class _ActorRLMemory(IRLMemoryWorker):
    def __init__(self, mq: RabbitMQManager, mq_capacity: int, client_id):
        self.mq = mq
        self.mq_capacity = mq_capacity
        self.client_id = client_id
        self.count = 0

    def add(self, *args) -> None:
        while True:
            qsize = self.mq.fetch_qsize_once(f"memory_{self.client_id}")
            if 0 <= qsize < self.mq_capacity:
                if self.mq.send_once(f"memory_{self.client_id}", args):
                    self.count += 1
                break

            # keepalive
            if self.mq.keepalive():
                if not self.mq.taskcheck_alive_task(self.client_id):
                    break

            time.sleep(1)

    def length(self) -> int:
        return self.count


class _ActorInterrupt(Callback):
    def __init__(
        self,
        mq: RabbitMQManager,
        client_id,
        parameter: RLParameter,
        actor_parameter_sync_interval_by_step: int,
    ) -> None:
        self.mq = mq
        self.client_id = client_id
        self.parameter = parameter
        self.actor_parameter_sync_interval_by_step = actor_parameter_sync_interval_by_step
        self.step = 0

    def on_episodes_begin(self, runner: srl.Runner):
        runner.state.sync_actor = 0

    def on_step_end(self, runner: srl.Runner):
        # --- sync params
        self.step += 1
        if self.step % self.actor_parameter_sync_interval_by_step != 0:
            return
        self.mq.create_fanout_queue_once_if_not_exists(f"parameter_{self.mq.uid}", "parameter")
        body = self.mq.recv_once_lastdata_and_purge(f"parameter_{self.mq.uid}")
        if body is not None:
            self.parameter.restore(body)
            runner.state.sync_actor += 1

    def intermediate_stop(self, runner) -> bool:
        # --- keepalive
        if self.mq.keepalive():
            if not self.mq.taskcheck_alive_task(self.client_id):
                return True
        return False


def _run_actor(mq: RabbitMQManager, mp_data: RunnerMPData, actor_id: int, client_id):
    mp_data.context.run_name = RunNameTypes.actor
    mp_data.context.actor_id = actor_id
    logger.info(f"actor_id={actor_id}")

    # --- runner
    runner = srl.Runner(mp_data.context.env_config, mp_data.context.rl_config, mp_data.config, mp_data.context)

    # --- memory
    mq_capacity = 1000  # TODO
    memory = _ActorRLMemory(mq, mq_capacity, client_id)

    # --- play
    runner.context.callbacks.append(
        _ActorInterrupt(
            mq,
            client_id,
            runner.make_parameter(),
            runner.context.actor_parameter_sync_interval_by_step,
        )
    )
    runner.context.disable_trainer = True
    runner.core_play(trainer_only=False, memory=cast(RLMemory, memory))


class ActorServerCallback:
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
    callbacks: List[ActorServerCallback] = [],
):
    mq = RabbitMQManager(host, port, username, password, virtual_host)
    mq.join("actor")
    mq.board_update({"status": "", "client": ""})

    print(f"wait actor: {mq.uid}")
    while True:
        time.sleep(1)
        if mq.keepalive():
            mq.health_check()

        try:
            # --- callback
            # [c.on_polling(channel) for c in callbacks]
            # for c in callbacks:
            #    if c.intermediate_stop(channel):
            #        break

            # --- board check
            client_id, mp_data, actor_id = mq.taskcheck_if_my_id_assigned()

            # --- actor loop
            if client_id != "":
                assert mp_data is not None
                logger.info(f"actor{mq.uid} start")
                _run_actor(mq, mp_data, actor_id, client_id)
                logger.info(f"actor{mq.uid} end")
                print(f"wait actor: {mq.uid}")

            # --- boardを更新
            mq.board_update({"status": "", "client": ""})

        except Exception:
            logger.warning(traceback.format_exc())
