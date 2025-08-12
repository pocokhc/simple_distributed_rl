import logging
import pickle
import queue
import sys
import time
import traceback
from typing import Callable, cast

from srl.base.context import RunContext
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.run import core_play
from srl.base.run.callback import RunCallback
from srl.base.run.core_play import RunStateActor
from srl.runner.distribution.server_manager import ServerManager

logger = logging.getLogger(__name__)


class _ActorRLMemoryInterceptor:
    def __init__(
        self,
        server: ServerManager,
        dist_queue_capacity: int,
        base_memory: RLMemory,
        uid,
        keepalive_interval: int,
    ):
        self.memory_sender = server.get_memory_sender()
        self.task = server.get_task_manager("actor", uid)

        self.dist_queue_capacity = dist_queue_capacity
        self.q_send_count = 0
        self.q = queue.Queue()
        self.actor_num = self.task.get_actor_num()
        self.serialize_funcs = {k: v[1] for k, v in base_memory.get_worker_funcs().items()}

        self.keepalive_interval = keepalive_interval
        self.keepalive_t0 = 0

    def length(self) -> int:
        return self.q_send_count

    def __getattr__(self, name: str) -> Callable:
        if name not in self.serialize_funcs:
            raise AttributeError(f"{name} is not a valid method")
        serialize_func = self.serialize_funcs[name]

        def wrapped(*args, **kwargs):
            t0 = time.time()
            while True:
                # --- server check
                remote_qsize = -1
                if not self.memory_sender.is_connected:
                    self.memory_sender.ping()
                if self.memory_sender.is_connected:
                    remote_qsize = self.memory_sender.memory_size()

                    # remote_qsizeが取得できない場合は受信と送信から予測
                    if remote_qsize < 0:
                        qsize = self.q_send_count * self.actor_num
                        q_recv_count = self.task.get_trainer("q_recv_count")
                        q_recv_count = 0 if q_recv_count == "" else int(q_recv_count)
                        remote_qsize = qsize - q_recv_count

                    # --- qが一定以下のみ送信
                    if remote_qsize < self.dist_queue_capacity:
                        try:
                            raw = serialize_func(*args, **kwargs)
                            raw = raw if isinstance(raw, tuple) else (raw,)
                            self.memory_sender.memory_send((name, raw))
                            self.q_send_count += 1
                        except Exception as e:
                            logger.error(e)
                        break

                # --- keepalive
                if time.time() - self.keepalive_t0 > self.keepalive_interval:
                    self.keepalive_t0 = time.time()
                    if self.task.is_finished():
                        break

                if time.time() - t0 > 9:
                    t0 = time.time()
                    s = f"capacity over, wait: local {self.q.qsize()}, remote_qsize {remote_qsize}"
                    print(s)
                    logger.info(s)
                    break

                time.sleep(1)

        return wrapped


class _ActorInterrupt(RunCallback):
    def __init__(
        self,
        server: ServerManager,
        parameter: RLParameter,
        actor_id: int,
        actor_parameter_sync_interval: int,
        uid,
        keepalive_interval: int,
    ) -> None:
        self.task = server.get_task_manager("actor", uid)
        self.actor_id = actor_id
        self.keepalive_interval = keepalive_interval

        self.parameter = parameter
        self.actor_parameter_sync_interval = actor_parameter_sync_interval
        self.t0 = time.time()

    def on_episodes_begin(self, context: RunContext, state: RunStateActor, **kwargs):
        state.sync_actor = 0
        self._keepalive_t0 = time.time()

    def on_step_end(self, context: RunContext, state: RunStateActor, **kwargs) -> bool:
        # --- sync params
        if time.time() - self.t0 > self.actor_parameter_sync_interval:
            if self.task.read_parameter(self.parameter):
                state.sync_actor += 1
                state.train_count = self.task.get_train_count()
            self.t0 = time.time()

        state.actor_send_q = state.memory.length()

        # --- keepalive
        if time.time() - self._keepalive_t0 > self.keepalive_interval:
            self.task.keepalive_actor(self.actor_id, state.total_step, state.memory.length())
            self._keepalive_t0 = time.time()
            if self.task.is_finished():
                return True
        return False

    def on_episodes_end(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        self.task.keepalive_actor(self.actor_id, state.total_step, state.memory.length())


def run_actor(server: ServerManager, actor_id: int, uid: str, keepalive_interval: int):
    task = server.get_task_manager("actor", uid=uid)
    context = None
    try:
        task_cfg = task.get_config()
        if task_cfg is None:
            return

        context = task_cfg.context
        context.run_name = "actor"
        context.actor_id = actor_id
        context.setup_device()
        env = context.env_config.make()

        # --- parameter
        parameter = context.rl_config.make_parameter(env=env)
        task.read_parameter(parameter)

        # --- memory
        memory = _ActorRLMemoryInterceptor(
            server,
            task_cfg.queue_capacity,
            context.rl_config.make_memory(),
            uid=uid,
            keepalive_interval=keepalive_interval,
        )

        # --- callback
        context.callbacks.append(
            _ActorInterrupt(
                server,
                parameter,
                actor_id,
                task_cfg.actor_parameter_sync_interval,
                uid=uid,
                keepalive_interval=keepalive_interval,
            )
        )

        # --- play
        context.training = True
        context.disable_trainer = True
        # context.max_episodes = 0
        context.max_memory = 0
        # context.max_steps = 0
        context.max_train_count = 0
        # context.timeout = 0
        workers, main_worker_idx = context.rl_config.make_workers(context.players, env, parameter, cast(RLMemory, memory))
        core_play.play(context, env, workers[main_worker_idx], workers=workers)

        task.unassign("actor", actor_id, reason="completed")
        print("actor end")
    finally:
        if (context is not None) and context.rl_config.use_update_parameter_from_worker():
            # actor0のみ送信
            if actor_id == 0:
                pass  # TODO


if __name__ == "__main__":
    redis_params, memory_params, actor_id, uid, keepalive_interval = pickle.loads(sys.stdin.buffer.read())
    print(f"--- start actor{actor_id} process")
    server = ServerManager(redis_params, memory_params)
    try:
        run_actor(server, actor_id, uid, keepalive_interval)
    except Exception:
        print(traceback.format_exc())
    finally:
        print(f"--- end actor{actor_id} process")
