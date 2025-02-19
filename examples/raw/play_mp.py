import ctypes
import multiprocessing as mp
import queue
from typing import Any, List

import srl
from srl.base.context import RunContext
from srl.base.env.config import EnvConfig
from srl.base.rl.config import RLConfig
from srl.base.rl.memory import IRLMemoryWorker, RLMemory
from srl.base.rl.parameter import RLParameter

# --- env & algorithm
from srl.envs import grid  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip


class _ActorRLMemory(IRLMemoryWorker):
    def __init__(self, remote_queue: queue.Queue, base_memory: IRLMemoryWorker):
        self.remote_queue = remote_queue
        self.base_memory = base_memory

    def add(self, *args) -> None:
        raw = self.base_memory.serialize_add_args(*args)
        self.remote_queue.put(raw)

    def length(self) -> int:
        return self.remote_queue.qsize()

    def serialize_add_args(self, *args):
        raise NotImplementedError("Unused")


def _run_actor(
    config,
    remote_queue: queue.Queue,
    remote_board: Any,
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
    parameter = rl_config.make_parameter(is_load=False)
    memory = rl_config.make_memory(is_load=False)
    remote_memory = _ActorRLMemory(remote_queue, memory)
    worker = rl_config.make_worker(env, parameter, remote_memory)
    env.setup(context)
    worker.setup(context)

    # episode loop
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
        if episode % config["actor_parameter_sync_interval_by_episode"] == 0:
            params = remote_board.value
            if params is not None:
                parameter.restore(params)

        if episode % 1000 == 0:
            print(f"actor{actor_id} : {episode} episode, {env.step_num} step, {env.episode_rewards} reward")
    worker.teardown()
    env.teardown()


def _run_trainer(
    config,
    parameter: RLParameter,
    memory: RLMemory,
    remote_queue: queue.Queue,
    remote_board: Any,
    train_end_signal: Any,
):
    rl_config: RLConfig = config["rl_config"]
    context: RunContext = config["context"]

    trainer = rl_config.make_trainer(parameter, memory)
    trainer.setup(context)

    train_count = 0
    recv_queue = 0
    while True:
        if train_end_signal.value:
            break

        if train_count >= context.max_train_count:
            break

        trainer.train()
        train_count = trainer.get_train_count()

        # recv memory
        if not remote_queue.empty():
            batch = remote_queue.get()
            batch = memory.deserialize_add_args(batch)
            memory.add(*batch)
            recv_queue += 1

        # send parameter
        if train_count % config["trainer_parameter_send_interval_by_train_count"] == 0:
            remote_board.set(parameter.backup())

        if train_count > 0 and train_count % 10000 == 0:
            print(f"trainer: {train_count} / {context.max_train_count}, {recv_queue=}")

    train_end_signal.value = True
    trainer.teardown()


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
        "actor_parameter_sync_interval_by_episode": 1,
    }

    # init
    env = env_config.make()
    rl_config.setup(env)

    # make parameter/memory
    parameter = rl_config.make_parameter()
    memory = rl_config.make_memory()

    # bug fix
    mp.set_start_method("spawn")

    # --- async
    with mp.Manager() as manager:
        # --- share values
        train_end_signal = mp.Value(ctypes.c_bool, False)
        remote_queue = manager.Queue()
        remote_board = manager.Value(ctypes.c_char_p, None)

        # --- actor
        actors_ps_list: List[mp.Process] = []
        for actor_id in range(context.actor_num):
            params = (
                config,
                remote_queue,
                remote_board,
                actor_id,
                train_end_signal,
            )
            ps = mp.Process(target=_run_actor, args=params)
            actors_ps_list.append(ps)
        # actor start
        [p.start() for p in actors_ps_list]

        # --- trainer start
        _run_trainer(
            config,
            parameter,
            memory,
            remote_queue,
            remote_board,
            train_end_signal,
        )

        # 強制終了
        [p.terminate() for p in actors_ps_list]

    # --------------------
    # rendering
    # --------------------
    context = RunContext(render_mode="terminal")
    worker = rl_config.make_worker(env, parameter)
    env.setup(context)
    worker.setup(context)

    env.reset()
    worker.reset(0)

    print("step 0")
    env.render()
    while not env.done:
        action = worker.policy()

        print(f"player {env.next_player}")
        worker.render()

        env.step(action)
        worker.on_step()

        print("--- turn {}, action {}, rewards: {}, done: {}, next player {}, info: {}, ".format(env.step_num, action, env.rewards, env.done, env.next_player, env.info))
        print("player {} info: {}".format(env.next_player, worker.info))
        env.render()

    print(f"step: {env.step_num}, reward: {env.episode_rewards}")
    env.teardown()
    worker.teardown()


if __name__ == "__main__":
    main()
