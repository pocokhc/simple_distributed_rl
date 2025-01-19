import multiprocessing as mp
import threading
from multiprocessing.managers import BaseManager
from typing import Any

import numpy as np

import srl
from srl.algorithms import ql
from srl.base.context import RunContext
from srl.base.env.env_run import EnvRun
from srl.base.rl.worker_run import WorkerRun


class Board:
    def __init__(self):
        self.params = None
        self.update_count = 0

    def write(self, params):
        self.params = params
        self.update_count += 1

    def get_update_count(self):
        return self.update_count

    def read(self):
        return self.params


class ServerState:
    def __init__(self) -> None:
        self.end_signal: bool = False
        self.run_actor: int = 0

    def set_end_signal(self, end_signal: bool):
        self.end_signal = end_signal

    def get_end_signal(self) -> bool:
        return self.end_signal

    def join_actor(self, max_actor: int) -> int:
        actor_id = self.run_actor % max_actor
        print(f"join actor: {actor_id}")
        self.run_actor += 1
        return actor_id


class MPManager(BaseManager):
    pass


def run_trainer(queue, ip, port):
    MPManager.register("get_config")
    MPManager.register("get_server_state")
    MPManager.register("RemoteMemory")
    MPManager.register("Board")
    MPManager.register("server_stop")

    manager: Any = MPManager(address=(ip, port), authkey=b"abracadabra")
    manager.connect()

    config = manager.get_config().copy()
    rl_config = config["rl_config"]
    context = config["context"]
    server_state: ServerState = manager.get_server_state()
    remote_board = manager.Board()
    remote_memory = manager.RemoteMemory()

    parameter = rl_config.make_parameter()
    trainer = rl_config.make_trainer(parameter, remote_memory)
    trainer.setup(context)

    train_count = 0
    while True:
        if server_state.get_end_signal():
            break

        if train_count >= context.max_train_count:
            break

        trainer.train()
        train_count = trainer.get_train_count()

        # send parameter
        if train_count % config["trainer_parameter_send_interval_by_train_count"] == 0:
            remote_board.write(parameter.backup())

        if train_count > 0 and train_count % 10000 == 0:
            print(f"{train_count} / 50000 train")

    server_state.set_end_signal(True)
    queue.put(parameter.backup())  # 学習結果を送信
    trainer.teardown()
    manager.server_stop()


def run_server(config):
    server_state = ServerState()
    board = Board()
    remote_memory = config["rl_config"].make_memory()
    MPManager.register("get_config", callable=lambda: config)
    MPManager.register("get_server_state", callable=lambda: server_state)
    MPManager.register("RemoteMemory", callable=lambda: remote_memory)
    MPManager.register("Board", callable=lambda: board)

    manager = MPManager(address=("", config["port"]), authkey=b"abracadabra")
    server: Any = manager.get_server()

    # add server stop function
    shutdown_timer = threading.Timer(1, lambda: server.stop_event.set())
    MPManager.register("server_stop", callable=lambda: shutdown_timer.start())

    print("--- server start ---")
    server.serve_forever()


def train(config):
    env = srl.make_env(config["env_config"])
    config["rl_config"].setup(env)

    # bug fix
    mp.set_start_method("spawn")

    # -- server & trainer
    queue = mp.Queue()
    ps_trainer = mp.Process(target=run_trainer, args=(queue, config["ip"], config["port"]))
    ps_server = mp.Process(target=run_server, args=(config,))

    # --- run
    ps_server.start()
    ps_trainer.start()
    ps_server.join()
    print("--- server end ---")

    # --- last param
    param = queue.get()
    parameter = rl_config.make_parameter()
    parameter.restore(param)

    return parameter


def _run_episode(env: EnvRun, worker: WorkerRun):
    env.reset()
    worker.reset(0)
    while not env.done:
        action = worker.policy()
        env.step(action)
        worker.on_step()
    return env.episode_rewards[0]


if __name__ == "__main__":
    # --- config
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()
    context = RunContext(
        actor_num=2,
        max_train_count=50000,
        distributed=True,
        training=True,
    )
    config = {
        "ip": "127.0.0.1",
        "port": 50000,
        "env_config": env_config,
        "rl_config": rl_config,
        "context": context,
        "trainer_parameter_send_interval_by_train_count": 100,
    }

    # --- remote train
    parameter = train(config)

    # --- evaluate
    context = RunContext()
    env = env_config.make()
    assert env.player_num == 1
    worker = srl.make_worker(rl_config, env, parameter)
    env.setup(context)
    worker.setup(context)

    reward_list = []
    for episode in range(100):
        reward = _run_episode(env, worker)
        reward_list.append(reward)
    print(f"Average reward for 100 episodes: {np.mean(reward_list):.5f}")

    env.teardown()
    worker.teardown()
