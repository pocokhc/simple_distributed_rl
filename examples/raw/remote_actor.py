from multiprocessing.managers import BaseManager
from typing import Any

import srl
from srl.base.context import RunContext
from srl.base.env.config import EnvConfig
from srl.base.env.env_run import EnvRun
from srl.base.rl.config import RLConfig
from srl.base.rl.memory import RLMemory
from srl.base.rl.worker_run import WorkerRun


def _run_episode(env: EnvRun, worker: WorkerRun):
    env.reset()
    worker.on_reset(0)
    while not env.done:
        action = worker.policy()
        env.step(action)
        worker.on_step()
    return env.step_num, env.episode_rewards[0]


class MPManager(BaseManager):
    pass


def main(ip: str, port: int):
    MPManager.register("get_config")
    MPManager.register("get_server_state")
    MPManager.register("RemoteMemory")
    MPManager.register("Board")

    manager: Any = MPManager(address=(ip, port), authkey=b"abracadabra")
    manager.connect()

    config = manager.get_config().copy()
    rl_config: RLConfig = config["rl_config"]
    env_config: EnvConfig = config["env_config"]
    context: RunContext = config["context"]
    server_state = manager.get_server_state()
    remote_board = manager.Board()
    remote_memory: RLMemory = manager.RemoteMemory()

    actor_id = server_state.join_actor(context.actor_num)
    rl_config.setup_from_actor(context.actor_num, actor_id)

    env = srl.make_env(env_config)
    assert env.player_num == 1
    parameter = srl.make_parameter(rl_config)
    worker = srl.make_worker(rl_config, env, parameter, remote_memory)

    env.setup(context)
    worker.on_start(context)

    print("--- actor start ---")
    prev_update_count = 0
    episode = 0
    while True:
        if server_state.get_end_signal():
            break

        step, reward = _run_episode(env, worker)
        episode += 1

        # --- sync parameter
        update_count = remote_board.get_update_count()
        if update_count != prev_update_count:
            prev_update_count = update_count
            params = remote_board.read()
            if params is not None:
                parameter.restore(params)

        if episode % 1000 == 0:
            print(f"{actor_id}: {episode} episode, {step} step, {reward} reward")

    print("--- actor end ---")


if __name__ == "__main__":
    ip = "127.0.0.1"
    port = 50000
    main(ip, port)
