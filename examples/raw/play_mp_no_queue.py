import ctypes
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from typing import Any, List, cast

import srl
from srl.base.context import RunContext
from srl.base.env.config import EnvConfig
from srl.base.rl.config import RLConfig
from srl.base.rl.memory import RLMemory
from srl.base.rl.registration import make_memory_class

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
    remote_memory: RLMemory,
    remote_board: Board,
    actor_id: int,
    train_end_signal: ctypes.c_bool,
):
    env_config: EnvConfig = config["env_config"]
    rl_config: RLConfig = config["rl_config"]
    context: RunContext = config["context"]
    rl_config.setup_from_actor(context.actor_num, actor_id)
    context.actor_id = actor_id

    # make instance
    env = env_config.make()
    parameter = rl_config.make_parameter()
    worker = rl_config.make_worker(env, parameter, remote_memory)
    env.setup(context)
    worker.setup(context)

    # episode loop
    prev_update_count = 0
    episode = 0
    while True:
        if train_end_signal.value:
            break

        # --- 1 episode
        env.reset()
        worker.reset(0)
        while not env.done:
            action = worker.policy()
            env.step(action)
            worker.on_step()
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
    worker.teardown()
    env.teardown()


def _run_trainer(
    config,
    remote_memory: RLMemory,
    remote_board: Board,
    train_end_signal: ctypes.c_bool,
):
    rl_config: RLConfig = config["rl_config"]
    context: RunContext = config["context"]

    parameter = rl_config.make_parameter()
    trainer = rl_config.make_trainer(parameter, remote_memory)
    trainer.setup(context)

    train_count = 0
    while True:
        if train_end_signal.value:
            break

        if train_count >= context.max_train_count:
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

    trainer.teardown()


class MPManager(BaseManager):
    pass


def main():
    # --- config
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()
    context = RunContext(
        actor_num=2,
        max_train_count=100000,
        distributed=True,
        training=True,
    )
    config = {
        "env_config": env_config,
        "rl_config": rl_config,
        "context": context,
        "trainer_parameter_send_interval_by_train_count": 100,
    }

    # init
    env = env_config.make()
    rl_config.setup(env)

    # bug fix
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        method = mp.get_start_method(allow_none=True)
        if method != "spawn":
            print("Start method is not 'spawn'. Current: " + str(method))

    # --- async
    MPManager.register("RemoteMemory", make_memory_class(rl_config))
    MPManager.register("Board", Board)

    with MPManager() as manager:
        manager = cast(Any, manager)

        # --- share values
        train_end_signal = mp.Value(ctypes.c_bool, False)
        remote_memory = manager.RemoteMemory(rl_config)
        remote_board = manager.Board()

        # --- actor
        actors_ps_list: List[mp.Process] = []
        for actor_id in range(context.actor_num):
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
        parameter = rl_config.make_parameter()
        params = remote_board.read()
        if params is not None:
            parameter.restore(params)

        # 強制終了
        [p.terminate() for p in actors_ps_list]
        trainer_ps.terminate()

    # --------------------
    # rendering
    # --------------------
    context = RunContext(env_render_mode="terminal", rl_render_mode="terminal")
    worker = rl_config.make_worker(env, parameter)
    env.setup(context)
    worker.setup(context)

    env.reset()
    worker.reset(0)

    print("step 0")
    action = None
    while not env.done:
        print("--- turn {}, action {}, rewards: {}, done: {}, next player {}, info: {}, ".format(env.step_num, action, env.rewards, env.done, env.next_player, env.info))
        print("player {} info: {}".format(env.next_player, worker.info))
        env.render()

        action = worker.policy()
        env.step(action)
        worker.on_step()

    print(f"\n--- turn: {env.step_num}, reward: {env.rewards[0]}, total reward: {env.episode_rewards[0]}, done reason: {env.done_reason}")
    env.render()
    env.teardown()
    worker.teardown()


if __name__ == "__main__":
    main()
