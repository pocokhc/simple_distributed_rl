import ctypes
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from typing import Any, Type, cast

import srl
from srl.base.rl.base import RLRemoteMemory
from srl.base.rl.registration import make_parameter, make_remote_memory, make_trainer

# --- env & algorithm
from srl.envs import grid  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip


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


def _run_actor(
    config,
    remote_memory: RLRemoteMemory,
    remote_board: Board,
    actor_id: int,
    train_end_signal: ctypes.c_bool,
):
    env_config = config["env_config"]
    rl_config = config["rl_config"]
    rl_config.set_config_by_actor(config["actor_num"], actor_id)

    # make instance
    env = srl.make_env(env_config)
    parameter = make_parameter(rl_config)
    workers = [srl.make_worker(rl_config, env, parameter, remote_memory, distributed=True, actor_id=actor_id)]

    prev_update_count = 0
    episode = 0

    # episode loop
    while True:
        if train_end_signal.value:
            break

        # --- 1 episode
        # reset
        env.reset()
        [w.on_reset(i, training=True) for i, w in enumerate(workers)]

        while not env.done:
            # action
            action = workers[env.next_player_index].policy()

            # step
            env.step(action)
            [w.on_step() for w in workers]

        episode += 1

        # --- sync parameter
        update_count = remote_board.get_update_count()
        if update_count != prev_update_count:
            prev_update_count = update_count
            params = remote_board.read()
            if params is not None:
                parameter.restore(params)

        if episode % 1000 == 0:
            print(f"{actor_id}: {episode} episode, {env.step_num} step, {env.episode_rewards} reward")


def _run_trainer(
    config,
    remote_memory: RLRemoteMemory,
    remote_board: Board,
    train_end_signal: ctypes.c_bool,
):
    rl_config = config["rl_config"]

    parameter = make_parameter(rl_config)
    trainer = make_trainer(rl_config, parameter, remote_memory)

    train_count = 0
    while True:
        if train_end_signal.value:
            break

        if train_count >= config["max_train_count"]:
            break

        trainer.train()
        train_count = trainer.get_train_count()

        # send parameter
        if train_count % config["trainer_parameter_send_interval_by_train_count"] == 0:
            remote_board.write(parameter.backup())

        if train_count > 0 and train_count % 10000 == 0:
            print(f"{train_count} / 100000 train")

    train_end_signal.value = True

    # 学習結果を送信
    remote_board.write(parameter.backup())


class MPManager(BaseManager):
    pass


def main():
    # --- config
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()
    actor_num = 2
    config = {
        "env_config": env_config,
        "rl_config": rl_config,
        "max_train_count": 100000,
        "trainer_parameter_send_interval_by_train_count": 100,
        "actor_num": actor_num,
    }

    # init
    env = srl.make_env(env_config)
    rl_config.reset(env)

    # bug fix
    mp.set_start_method("spawn")

    # --- async
    MPManager.register("RemoteMemory", cast(Type[RLRemoteMemory], make_remote_memory(rl_config, return_class=True)))
    MPManager.register("Board", Board)

    with MPManager() as manager:
        manager = cast(Any, manager)

        # --- share values
        train_end_signal = mp.Value(ctypes.c_bool, False)
        remote_memory = manager.RemoteMemory(rl_config)
        remote_board = manager.Board()

        # --- actor
        actors_ps_list = []
        for actor_id in range(actor_num):
            params = (
                config,
                remote_memory,
                remote_board,
                actor_id,
                train_end_signal,
            )
            ps = mp.Process(target=_run_actor, args=params)
            actors_ps_list.append(ps)

        # --- trainer
        params = (
            config,
            remote_memory,
            remote_board,
            train_end_signal,
        )
        trainer_ps = mp.Process(target=_run_trainer, args=params)

        # --- start
        [p.start() for p in actors_ps_list]
        trainer_ps.start()
        trainer_ps.join()

        # 学習後の結果
        parameter = make_parameter(rl_config)
        params = remote_board.read()
        if params is not None:
            parameter.restore(params)

        # 強制終了
        [p.terminate() for p in actors_ps_list]
        trainer_ps.terminate()

    # --------------------
    # rendering
    # --------------------
    workers = [srl.make_worker(rl_config, env, parameter)]

    env.reset(render_mode="terminal")
    [w.on_reset(i, training=False, render_mode="terminal") for i, w in enumerate(workers)]

    print("step 0")
    env.render()

    while not env.done:
        action = workers[env.next_player_index].policy()

        print(f"player {env.next_player_index}")
        workers[env.next_player_index].render()

        env.step(action)
        [w.on_step() for w in workers]

        print(
            "--- turn {}, action {}, rewards: {}, done: {}, next player {}, info: {}, ".format(
                env.step_num, action, env.step_rewards, env.done, env.next_player_index, env.info
            )
        )
        print("player {} info: {}".format(env.next_player_index, workers[env.next_player_index].info))
        env.render()

    print(f"step: {env.step_num}, reward: {env.episode_rewards}")


if __name__ == "__main__":
    main()
