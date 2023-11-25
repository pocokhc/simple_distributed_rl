import os

import srl
from srl.algorithms import ql
from srl.runner.distribution import RedisParameters, TaskManager
from srl.utils import common

redis_params = RedisParameters(host="localhost")
checkpoint_dir = os.path.join(os.path.dirname(__file__), "_checkpoint")
history_dir = os.path.join(os.path.dirname(__file__), "_history")


def _create_runner():
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()
    return srl.Runner(env_config, rl_config)


def create_task():
    runner = _create_runner()
    runner.train_distribution_start(
        redis_params,
        timeout=60,
    )


def wait_task():
    task_manager = TaskManager(redis_params)
    task_manager.train_wait(
        checkpoint_save_dir=checkpoint_dir,
        checkpoint_interval=10,
        history_save_dir=history_dir,
    )


def eval_task():
    task_manager = TaskManager(redis_params)
    runner = task_manager.create_runner()
    if runner is None:
        print("Task not found.")
        return
    print(runner.evaluate())


def eval_checkpoint():
    runner = _create_runner()
    runner.load_checkpoint(checkpoint_dir)
    print(runner.evaluate())


def view_history():
    history = srl.Runner.load_history(history_dir)
    history.plot(ylabel_left=["eval_reward0"])


if __name__ == "__main__":
    common.logger_print()
    create_task()
    wait_task()
    eval_task()
    eval_checkpoint()
    view_history()
