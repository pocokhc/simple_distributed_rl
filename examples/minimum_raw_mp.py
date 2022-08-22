import ctypes
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from typing import Optional

import srl
from srl.base.env.base import EnvRun
from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory
from srl.base.rl.registration import make_parameter, make_remote_memory, make_trainer


def _run_episode(
    env: EnvRun,
    rl_config: RLConfig,
    parameter: RLParameter,
    remote_memory: Optional[RLRemoteMemory],
    training: bool,
    distributed: bool,
    rendering: bool = False,
):

    workers = [srl.rl.make_worker(rl_config, parameter, remote_memory, training=training, distributed=distributed)]

    # --- reset
    env.reset()
    [w.on_reset(env, i) for i, w in enumerate(workers)]

    if rendering:
        print("step 0")
        env.render()

    while not env.done:

        # action
        action = workers[env.next_player_index].policy(env)

        if rendering:
            print(f"player {env.next_player_index}")
            workers[env.next_player_index].render(env)

        # step
        env_info = env.step(action)
        worker_infos = [w.on_step(env) for w in workers]

        # --- render
        if rendering:
            print(
                "turn {}, action {}, rewards: {}, done: {}, next player {}, info: {}, ".format(
                    env.step_num, action, env.step_rewards, env.done, env.next_player_index, env_info
                )
            )
            print("player {} info: {}".format(env.next_player_index, worker_infos[env.next_player_index]))
            env.render()

    return env.step_num, env.episode_rewards


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

    env = srl.envs.make(env_config)
    parameter = make_parameter(rl_config)

    prev_update_count = 0
    episode = 0

    # episode loop
    while True:
        if train_end_signal.value:
            break

        step, reward = _run_episode(
            env,
            rl_config,
            parameter,
            remote_memory,
            training=True,
            distributed=True,
        )
        episode += 1

        # sync parameter
        update_count = remote_board.get_update_count()
        if update_count != prev_update_count:
            prev_update_count = update_count
            params = remote_board.read()
            if params is not None:
                parameter.restore(params)

        if episode % 1000 == 0:
            print(f"{actor_id}: {episode} episode, {step} step, {reward} reward")


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
    env_config = srl.envs.Config("Grid")
    rl_config = srl.rl.ql.Config()
    actor_num = 2
    config = {
        "env_config": env_config,
        "rl_config": rl_config,
        "max_train_count": 100000,
        "trainer_parameter_send_interval_by_train_count": 100,
        "actor_num": actor_num,
    }

    # init
    env = srl.envs.make(env_config)
    rl_config.reset_config(env)

    # --- async
    MPManager.register("RemoteMemory", make_remote_memory(rl_config, return_class=True))
    MPManager.register("Board", Board)

    with MPManager() as manager:
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

    # --- rendering
    step, reward = _run_episode(
        env,
        rl_config,
        parameter,
        None,
        training=False,
        distributed=False,
        rendering=True,
    )
    print(f"step: {step}, reward: {reward}")


if __name__ == "__main__":
    main()
