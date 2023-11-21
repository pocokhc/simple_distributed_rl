import time

import srl
from srl.algorithms import ql
from srl.runner.distribution import RedisParameters, TaskManager
from srl.utils import common

redis_params = RedisParameters(host="localhost")


def create_task():
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()
    runner = srl.Runner(env_config, rl_config)

    runner.train_distribution_start(
        redis_params,
        max_train_count=1000,
    )


def wait_task():
    manager = TaskManager(redis_params)

    while True:
        print("wait task...")
        time.sleep(10)

        runner = manager.create_runner()
        if runner is None:
            print("Task not found.")
            break
        print(runner.evaluate())

        if manager.is_finished():
            break


def eval_task():
    manager = TaskManager(redis_params)
    runner = manager.create_runner()
    if runner is None:
        print("Task not found.")
        return
    print(runner.evaluate())


if __name__ == "__main__":
    common.logger_print()
    create_task()
    wait_task()
    eval_task()