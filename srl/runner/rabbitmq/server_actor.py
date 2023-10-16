import logging
import pickle
import time
import traceback
import uuid
from typing import cast

import pika
from pika.adapters.blocking_connection import BlockingChannel

import srl
from srl.base.rl.base import IRLMemoryWorker, RLMemory, RLParameter
from srl.base.run.data import RunNameTypes
from srl.runner.callback import Callback
from srl.runner.runner import RunnerMPData

logger = logging.getLogger(__name__)


class _ActorRLMemory(IRLMemoryWorker):
    def __init__(self, channel):
        self.channel = channel
        self.count = 0

    def add(self, *args) -> None:
        _q_size = self.channel.queue_declare(queue="memory", passive=True).method.message_count

        # 上限
        if _q_size < 1000:
            self.channel.basic_publish(exchange="", routing_key="memory", body=pickle.dumps(args))
            self.count += 1

    def length(self) -> int:
        return self.count


class _ActorInterrupt(Callback):
    def __init__(
        self,
        channel: BlockingChannel,
        parameter: RLParameter,
        actor_parameter_sync_interval_by_step: int,
        uid: uuid.UUID,
    ) -> None:
        self.channel = channel
        self.parameter = parameter
        self.actor_parameter_sync_interval_by_step = actor_parameter_sync_interval_by_step
        self.step = 0
        self.q_param_name = f"actor{uid}_parameter"
        self.q_end_name = f"actor{uid}_end"

    def on_episodes_begin(self, runner: srl.Runner):
        runner.state.sync_actor = 0

    def on_step_end(self, runner: srl.Runner):
        self.step += 1
        if self.step % self.actor_parameter_sync_interval_by_step != 0:
            return

        # 最新のメッセージを取り出し、残りは破棄する
        method_frame, header_frame, body = self.channel.basic_get(queue=self.q_param_name, auto_ack=True)
        if body is None:
            return
        self.channel.queue_purge(queue=self.q_param_name)
        self.parameter.restore(pickle.loads(body))
        runner.state.sync_actor += 1

    def intermediate_stop(self, runner) -> bool:
        method_frame, header_frame, body = self.channel.basic_get(queue=self.q_end_name, auto_ack=True)
        if body is not None:
            return True
        return False


def _run_actor(channel: BlockingChannel, mp_data: RunnerMPData, uid: uuid.UUID):
    # --- actor id ---
    # 最大10秒待つ、10秒たっても取得できない場合は0で進める
    actor_id = 0
    for _ in range(10):
        method_frame, header_frame, body = channel.basic_get(queue="actor", auto_ack=True)
        if body is not None:
            actor_id = int(body)
            break
        time.sleep(1)
    # ----------------

    mp_data.context.run_name = RunNameTypes.actor
    mp_data.context.actor_id = actor_id
    logger.info(f"actor_id={actor_id}")

    # --- runner
    runner = srl.Runner(mp_data.env_config, mp_data.rl_config, mp_data.config, mp_data.context)

    # --- memory
    memory = _ActorRLMemory(channel)

    # --- play
    runner.context.callbacks.append(
        _ActorInterrupt(
            channel,
            runner.make_parameter(),
            runner.context.actor_parameter_sync_interval_by_step,
            uid,
        )
    )
    runner.context.disable_trainer = True
    runner.core_play(trainer_only=False, memory=cast(RLMemory, memory))


def run_forever(
    host: str,
    port: int = 5672,
    user: str = "guest",
    password: str = "guest",
):
    uid = uuid.uuid4()
    credentials = pika.PlainCredentials(user, password)
    with pika.BlockingConnection(pika.ConnectionParameters(host, port, credentials=credentials)) as connection:
        channel = connection.channel()

        # start
        q_start_name = f"actor{uid}_start"
        channel.exchange_declare(exchange="start", exchange_type="fanout")
        channel.queue_declare(queue=q_start_name, auto_delete=True)
        channel.queue_bind(exchange="start", queue=q_start_name)

        # end
        q_end_name = f"actor{uid}_end"
        channel.exchange_declare(exchange="end", exchange_type="fanout")
        channel.queue_declare(queue=q_end_name, auto_delete=True)
        channel.queue_bind(exchange="end", queue=q_end_name)

        # parameter
        q_param_name = f"actor{uid}_parameter"
        channel.exchange_declare(exchange="parameter", exchange_type="fanout")
        channel.queue_declare(queue=q_param_name, auto_delete=True)
        channel.queue_bind(exchange="parameter", queue=q_param_name)

        try:
            print(f"wait actor: {uid}")
            while True:
                time.sleep(1)  # polling
                method_frame, header_frame, body = channel.basic_get(queue=q_start_name, auto_ack=True)
                if body is None:
                    continue
                try:
                    logger.info(f"actor{uid} start")
                    body = pickle.loads(body)
                    channel.queue_purge(queue=q_end_name)
                    channel.queue_purge(queue=q_param_name)

                    _run_actor(channel, body, uid)

                    channel.queue_purge(queue=q_start_name)
                except Exception:
                    logger.warning(traceback.format_exc())

                channel.basic_publish(exchange="end", routing_key="", body="STOP")
                logger.info(f"actor{uid} end")
                print(f"wait actor: {uid}")

        finally:
            channel.queue_delete(q_start_name)
            channel.queue_delete(q_end_name)
            channel.queue_delete(q_param_name)
