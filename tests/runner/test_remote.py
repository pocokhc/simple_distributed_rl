import multiprocessing as mp
import time

import numpy as np

import srl
from srl.algorithms import ql
from srl.utils import common

common.logger_print()


def run_server():
    runner = srl.Runner("Grid", ql.Config())
    runner.set_history(write_file=True)
    runner.set_save_dir("tmp_test")
    runner.train_remote(max_train_count=10_000)

    # eval
    rewards = runner.evaluate()
    rewards = np.mean(rewards)
    print(rewards)
    assert rewards > 0.5


def run_client():
    from srl.runner.core_remote import run_actor

    run_actor("127.0.0.1", 50000)


def test_train():
    server = mp.Process(target=run_server)
    client = mp.Process(target=run_client)

    server.start()
    time.sleep(2)
    client.start()

    while True:
        time.sleep(1)  # polling time

        if not server.is_alive():
            break
        if not client.is_alive():
            break

    server.join(timeout=10)

    assert server.exitcode == 0
    assert client.exitcode == 0
