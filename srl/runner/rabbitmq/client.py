import logging
import pickle
import time

import pika

import srl
from srl.utils import common

logger = logging.getLogger(__name__)

common.logger_print()


def run(
    runner: srl.Runner,
    host: str,
    port: int = 5672,
    user: str = "guest",
    password: str = "guest",
):
    parameter = runner.make_parameter()
    credentials = pika.PlainCredentials(user, password)
    with pika.BlockingConnection(pika.ConnectionParameters(host, port, credentials=credentials)) as connection:
        channel = connection.channel()

        channel.queue_declare(queue="last_parameter")

        # --- actor id
        channel.queue_declare(queue="actor")
        channel.queue_purge(queue="actor")
        for n in range(runner.context.actor_num):
            channel.basic_publish(exchange="", routing_key="actor", body=str(n))

        # --- send start
        body = runner.create_mp_data()
        channel.basic_publish(exchange="start", routing_key="", body=pickle.dumps(body))

        try:
            print("wait main")
            while True:
                time.sleep(1)
                method_frame, header_frame, body = channel.basic_get(queue="last_parameter", auto_ack=True)
                if body is not None:
                    parameter.restore(pickle.loads(body))
                    print("main end")
                    break
        finally:
            channel.queue_delete("actor")
            channel.queue_delete("last_parameter")
