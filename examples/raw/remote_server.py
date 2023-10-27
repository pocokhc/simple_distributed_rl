import multiprocessing as mp
import threading
from multiprocessing.managers import BaseManager
from typing import Any

import numpy as np

import srl
from srl.algorithms import ql
from srl.base.env.config import EnvConfig
from srl.base.rl.base import RLParameter
from srl.base.rl.config import RLConfig


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


def run_trainer(queue, train_config):
    MPManager.register("get_train_config")
    MPManager.register("get_rl_config")
    MPManager.register("get_server_state")
    MPManager.register("RemoteMemory")
    MPManager.register("Board")
    MPManager.register("server_stop")

    manager: Any = MPManager(address=(train_config["ip"], train_config["port"]), authkey=b"abracadabra")
    manager.connect()

    train_config = manager.get_train_config()
    rl_config = manager.get_rl_config().copy()
    server_state: ServerState = manager.get_server_state()
    remote_board = manager.Board()
    remote_memory = manager.RemoteMemory()

    parameter = srl.make_parameter(rl_config)
    trainer = srl.make_trainer(rl_config, parameter, remote_memory)

    train_count = 0
    while True:
        if server_state.get_end_signal():
            break

        if train_count >= train_config.get("max_train_count"):
            break

        trainer.train()
        train_count = trainer.get_train_count()

        # send parameter
        if train_count % train_config.get("trainer_parameter_send_interval_by_train_count") == 0:
            remote_board.write(parameter.backup())

        if train_count > 0 and train_count % 10000 == 0:
            print(f"{train_count} / 50000 train")

    server_state.set_end_signal(True)
    queue.put(parameter.backup())  # 学習結果を送信
    manager.server_stop()


def run_server(train_config, env_config: EnvConfig, rl_config: RLConfig):
    env = srl.make_env(env_config)
    rl_config.setup(env)
    server_state = ServerState()
    board = Board()
    remote_memory = srl.make_memory(rl_config)
    MPManager.register("get_train_config", callable=lambda: train_config)
    MPManager.register("get_rl_config", callable=lambda: rl_config)
    MPManager.register("get_env_config", callable=lambda: env_config)
    MPManager.register("get_server_state", callable=lambda: server_state)
    MPManager.register("RemoteMemory", callable=lambda: remote_memory)
    MPManager.register("Board", callable=lambda: board)

    manager = MPManager(address=("", 50000), authkey=b"abracadabra")
    server: Any = manager.get_server()

    # add server stop function
    shutdown_timer = threading.Timer(1, lambda: server.stop_event.set())
    MPManager.register("server_stop", callable=lambda: shutdown_timer.start())

    print("--- server start ---")
    server.serve_forever()


def train(train_config, env_config: EnvConfig, rl_config: RLConfig):
    env = srl.make_env(env_config)
    rl_config.setup(env)

    # bug fix
    mp.set_start_method("spawn")

    # -- server & trainer
    queue = mp.Queue()
    ps_trainer = mp.Process(target=run_trainer, args=(queue, train_config))
    ps_server = mp.Process(target=run_server, args=(train_config, env_config, rl_config))

    # --- run
    ps_server.start()
    ps_trainer.start()
    ps_server.join()
    print("--- server end ---")

    # --- last param
    param = queue.get()
    parameter = srl.make_parameter(rl_config)
    parameter.restore(param)

    return parameter


def _run_episode(env_config: EnvConfig, rl_config: RLConfig, parameter: RLParameter):
    env = srl.make_env(env_config)
    assert env.player_num == 1
    worker = srl.make_worker(rl_config, env, parameter)

    env.reset()
    worker.on_reset(0, training=False)

    while not env.done:
        action = worker.policy()
        env.step(action)
        worker.on_step()

    return env.episode_rewards[0]


if __name__ == "__main__":
    # --- config
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()
    train_config = {
        "ip": "127.0.0.1",
        "port": 50000,
        "max_train_count": 50000,
        "trainer_parameter_send_interval_by_train_count": 100,
        "max_actor": 2,
    }

    # --- remote train
    parameter = train(train_config, env_config, rl_config)

    # --- evaluate
    reward_list = []
    for episode in range(100):
        reward = _run_episode(env_config, rl_config, parameter)
        reward_list.append(reward)
    print(f"Average reward for 100 episodes: {np.mean(reward_list):.5f}")
