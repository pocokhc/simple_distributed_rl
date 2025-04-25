import ctypes
import multiprocessing as mp
import pickle
import queue
from typing import Any, Callable, List, cast

import srl
from srl.base.context import RunContext
from srl.base.env.config import EnvConfig
from srl.base.rl.config import RLConfig
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter

# --- env & algorithm
from srl.envs import grid  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip


class _ActorRLMemoryInterceptor:
    def __init__(self, remote_queue: queue.Queue, base_memory: RLMemory):
        self.remote_queue = remote_queue
        self.base_memory = base_memory

        # 登録されている関数を取得
        self.serialize_funcs = {k: v[1] for k, v in base_memory.get_worker_funcs().items()}

    def __getattr__(self, name: str) -> Callable:
        if name not in self.serialize_funcs:
            raise AttributeError(f"{name} is not a valid method")
        serialize_func = self.serialize_funcs[name]

        # 登録されている関数に割り込んでデータを送信
        def wrapped(*args, **kwargs):
            raw = serialize_func(*args, **kwargs)
            raw = raw if isinstance(raw, tuple) else (raw,)
            self.remote_queue.put((name, raw))

        return wrapped


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
    parameter = rl_config.make_parameter()
    memory = rl_config.make_memory()
    remote_memory = cast(RLMemory, _ActorRLMemoryInterceptor(remote_queue, memory))
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


def _run_memory(
    config,
    queue_act_to_mem: mp.Queue,
    queue_mem_to_train: mp.Queue,
    qsize_mem_to_train_list: list,
    queue_train_to_mem: mp.Queue,
    train_end_signal: Any,
):
    rl_config: RLConfig = config["rl_config"]
    memory = rl_config.make_memory()

    # worker -> memory
    worker_funcs = {k: v[0] for k, v in memory.get_worker_funcs().items()}

    trainer_recv_funcs = memory.get_trainer_recv_funcs()  # memory -> trainer
    trainer_send_funcs = memory.get_trainer_send_funcs()  # trainer -> memory

    send_count = 0
    while not train_end_signal.value:
        # worker -> memory
        if not queue_act_to_mem.empty():
            name, raw = queue_act_to_mem.get()
            worker_funcs[name](*raw, serialized=True)

        # memory -> trainer
        for i, trainer_recv_func in enumerate(trainer_recv_funcs):
            if qsize_mem_to_train_list[i].value < 10:
                batch = trainer_recv_func()
                if batch is not None:
                    queue_mem_to_train.put((i, pickle.dumps(batch)))
                    with qsize_mem_to_train_list[i].get_lock():
                        qsize_mem_to_train_list[i].value += 1
                    send_count += 1
                    if send_count > 0 and send_count % 5000 == 0:
                        print(f"memory : {send_count} send")

        # trainer -> memory
        if not queue_train_to_mem.empty():
            name, raw = queue_train_to_mem.get()
            args, kwargs = pickle.loads(raw)
            trainer_send_funcs[name](*args, **kwargs)


class _TrainerRLMemoryInterceptor:
    def __init__(
        self,
        qsize_mem_to_train_list: list,
        queue_train_to_mem: mp.Queue,
        base_memory: RLMemory,
    ):
        self.qsize_mem_to_train_list = qsize_mem_to_train_list
        self.base_memory = base_memory

        # --- memory -> trainer
        self.buffers = [[] for _ in range(len(base_memory.get_trainer_recv_funcs()))]
        self.trainer_recv_funcs = {
            func.__name__: i  #
            for i, func in enumerate(base_memory.get_trainer_recv_funcs())
        }

        # --- trainer -> memory
        self.queue_train_to_mem = queue_train_to_mem
        self.trainer_send_funcs = base_memory.get_trainer_send_funcs()

    def __getattr__(self, name: str) -> Callable:
        # --- memory -> trainer
        if name in self.trainer_recv_funcs:
            # memoryから受信したデータを代わりに返す
            idx = self.trainer_recv_funcs[name]
            if len(self.buffers[idx]) == 0:
                # データがない場合はNoneを返す
                return lambda: None
            with self.qsize_mem_to_train_list[idx].get_lock():
                self.qsize_mem_to_train_list[idx].value -= 1
            return lambda: self.buffers[idx].pop()

        # --- trainer -> memory
        if name in self.trainer_send_funcs:

            def wrapped(*args, **kwargs):
                self.queue_train_to_mem.put((name, pickle.dumps((args, kwargs))))

            return wrapped

        raise AttributeError(f"{name} is not a valid method")


def _run_trainer(
    config,
    parameter: RLParameter,
    queue_mem_to_train: mp.Queue,
    qsize_mem_to_train_list: list,
    queue_train_to_mem: mp.Queue,
    remote_board: Any,
    train_end_signal: Any,
):
    rl_config: RLConfig = config["rl_config"]
    context: RunContext = config["context"]

    memory = _TrainerRLMemoryInterceptor(
        qsize_mem_to_train_list,
        queue_train_to_mem,
        rl_config.make_memory(),
    )
    trainer = rl_config.make_trainer(parameter, cast(RLMemory, memory))
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
        if not queue_mem_to_train.empty():
            idx, raw = queue_mem_to_train.get()
            memory.buffers[idx].append(pickle.loads(raw))
            recv_queue += 1

        # send parameter
        if train_count % config["trainer_parameter_send_interval_by_train_count"] == 0:
            remote_board.set(parameter.backup())

        if train_count > 0 and train_count % 5000 == 0:
            print(f"trainer: {train_count} / {context.max_train_count}, {recv_queue=}")

    train_end_signal.value = True
    trainer.teardown()


def main():
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

    # bug fix
    if mp.get_start_method() != "spawn":
        mp.set_start_method("spawn")

    # --- async
    with mp.Manager() as manager:
        # --- share values
        train_end_signal = mp.Value(ctypes.c_bool, False)
        queue_act_to_mem = mp.Queue()
        queue_mem_to_train = mp.Queue()
        qsize_mem_to_train_list = [
            mp.Value(ctypes.c_int, 0)  #
            for _ in range(len(rl_config.make_memory().get_trainer_recv_funcs()))
        ]
        queue_train_to_mem = mp.Queue()
        remote_board = manager.Value(ctypes.c_char_p, None)

        # --- actor
        actors_ps_list: List[mp.Process] = []
        for actor_id in range(context.actor_num):
            params = (
                config,
                queue_act_to_mem,
                remote_board,
                actor_id,
                train_end_signal,
            )
            ps = mp.Process(target=_run_actor, args=params)
            actors_ps_list.append(ps)
        # actor start
        [p.start() for p in actors_ps_list]

        # --- memory
        memory_ps = mp.Process(
            target=_run_memory,
            args=(
                config,
                queue_act_to_mem,
                queue_mem_to_train,
                qsize_mem_to_train_list,
                queue_train_to_mem,
                train_end_signal,
            ),
        )
        memory_ps.start()

        # --- trainer start
        _run_trainer(
            config,
            parameter,
            queue_mem_to_train,
            qsize_mem_to_train_list,
            queue_train_to_mem,
            remote_board,
            train_end_signal,
        )

        # 強制終了
        [p.terminate() for p in actors_ps_list]
        memory_ps.terminate()

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
        action = worker.policy()

        print("--- turn {}, action {}, rewards: {}, done: {}, next player {}, info: {}, ".format(env.step_num, action, env.rewards, env.done, env.next_player, env.info))
        print("player {} info: {}".format(env.next_player, worker.info))
        env.render()
        worker.render()

        env.step(action)
        worker.on_step()

    print(f"\n--- turn: {env.step_num}, reward: {env.rewards[0]}, total reward: {env.episode_rewards[0]}, done reason: {env.done_reason}")
    env.render()
    env.teardown()
    worker.teardown()


if __name__ == "__main__":
    main()
