from multiprocessing.managers import BaseManager
from typing import Any, Optional

import srl
from srl.base.env.base import EnvRun
from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory

from srl.envs import grid  # isort: skip # noqa F401


def _run_episode(
    env: EnvRun,
    rl_config: RLConfig,
    parameter: RLParameter,
    remote_memory: Optional[RLRemoteMemory],
):
    assert env.player_num == 1
    worker = srl.make_worker(rl_config, parameter, remote_memory, training=True, distributed=True)

    env.reset()
    worker.on_reset(env)

    while not env.done:
        action = worker.policy(env)
        env.step(action)
        worker.on_step(env)

    return env.step_num, env.episode_rewards[0]


class MPManager(BaseManager):
    pass


def main(ip: str, port: int):
    MPManager.register("get_train_config")
    MPManager.register("get_rl_config")
    MPManager.register("get_env_config")
    MPManager.register("get_server_state")
    MPManager.register("RemoteMemory")
    MPManager.register("Board")

    manager: Any = MPManager(address=(ip, port), authkey=b"abracadabra")
    manager.connect()

    train_config = manager.get_train_config()
    rl_config = manager.get_rl_config().copy()
    env_config = manager.get_env_config().copy()
    server_state = manager.get_server_state()
    remote_board = manager.Board()
    remote_memory = manager.RemoteMemory()

    max_actor = train_config.get("max_actor")
    actor_id = server_state.join_actor(max_actor)
    rl_config.set_config_by_actor(max_actor, actor_id)

    env = srl.make_env(env_config)
    parameter = srl.make_parameter(rl_config)

    prev_update_count = 0
    episode = 0

    print("--- actor start ---")
    while True:
        if server_state.get_end_signal():
            break

        step, reward = _run_episode(env, rl_config, parameter, remote_memory)
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
