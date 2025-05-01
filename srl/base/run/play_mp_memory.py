import ctypes
import logging
import multiprocessing as mp
import pickle
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing import sharedctypes
from typing import Any, Callable, List, cast

from srl.base.context import RunContext, RunNameTypes
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.run import core_play, core_train_only
from srl.base.run.callback import RunCallback
from srl.base.run.core_play import RunStateActor
from srl.base.run.core_train_only import RunStateTrainer

# from multiprocessing.managers import ValueProxy # ValueProxy[bool] -> ctypes.c_bool

logger = logging.getLogger(__name__)


@dataclass
class MpConfig:
    context: RunContext
    callbacks: List[RunCallback] = field(default_factory=list)

    queue_capacity: int = 1000
    trainer_parameter_send_interval: float = 1  # sec
    actor_parameter_sync_interval: float = 1  # sec
    train_to_mem_queue_capacity: int = 100
    mem_to_train_queue_capacity: int = 10


# --------------------
# actor
# --------------------
class _ActorRLMemoryInterceptor:
    def __init__(
        self,
        queue_act_to_mem: mp.Queue,
        qsize_act_to_mem: sharedctypes.Synchronized,
        end_signal: ctypes.c_bool,
        actor_id: int,
        base_memory: RLMemory,
        cfg: MpConfig,
    ):
        self.queue_act_to_mem = queue_act_to_mem
        self.qsize_act_to_mem = qsize_act_to_mem
        self.end_signal = end_signal
        self.actor_id = actor_id
        self.serialize_funcs = {k: v[1] for k, v in base_memory.get_worker_funcs().items()}
        self.cfg = cfg
        self.q_send = 0
        self.t0_health = time.time()  # [2]

    def length(self) -> int:
        return self.qsize_act_to_mem.value

    def __getattr__(self, name: str) -> Callable:
        if name not in self.serialize_funcs:
            raise AttributeError(f"{name} is not a valid method")
        serialize_func = self.serialize_funcs[name]

        def wrapped(*args, **kwargs):
            t0 = time.time()
            while True:
                if self.end_signal.value:
                    break

                if self.qsize_act_to_mem.value < self.cfg.queue_capacity:
                    raw = serialize_func(*args, **kwargs)
                    # 引数で展開させるためにtupleにする、なので戻り値1つのtupleのみできない
                    raw = raw if isinstance(raw, tuple) else (raw,)
                    self.queue_act_to_mem.put((name, raw))
                    with self.qsize_act_to_mem.get_lock():
                        self.qsize_act_to_mem.value += 1
                    self.q_send += 1
                    break

                # [2] 親プロセスの生存確認
                if time.time() - self.t0_health > 5:
                    pparent = mp.parent_process()
                    assert pparent is not None
                    if not pparent.is_alive():
                        self.end_signal.value = True
                        logger.info(f"[actor{self.actor_id}] end_signal=True (parent process dead)")
                        break
                    self.t0_health = time.time()

                if time.time() - t0 > 9:
                    t0 = time.time()
                    s = f"[actor{self.actor_id}] The queue is full so I will wait. {self.qsize_act_to_mem.value}"
                    print(s)
                    logger.info(s)
                    break  # 終了条件用に1step進める
                time.sleep(1)

        return wrapped


class _ActorInterrupt(RunCallback):
    def __init__(
        self,
        remote_board: Any,
        parameter: RLParameter,
        end_signal: ctypes.c_bool,
        cfg: MpConfig,
    ) -> None:
        self.remote_board = remote_board
        self.parameter = parameter
        self.end_signal = end_signal
        self.cfg = cfg

        self.t0 = time.time()
        self.t0_health = time.time()  # [2]

    def on_episodes_begin(self, context: RunContext, state: RunStateActor, **kwargs):
        state.sync_actor = 0

    def on_step_end(self, context: RunContext, state: RunStateActor, **kwargs) -> bool:
        # [2] 親プロセスの生存確認
        if time.time() - self.t0_health > 5:
            pparent = mp.parent_process()
            assert pparent is not None
            if not pparent.is_alive():
                self.end_signal.value = True
                logger.info(f"[actor{context.actor_id}] end_signal=True (parent process dead)")
                return True
            self.t0_health = time.time()

        if time.time() - self.t0 < self.cfg.actor_parameter_sync_interval:
            return self.end_signal.value
        state.actor_send_q = cast(_ActorRLMemoryInterceptor, state.memory).q_send

        self.t0 = time.time()
        dat = self.remote_board.value
        if dat is None:
            return self.end_signal.value
        train_count, params = pickle.loads(dat)
        if params is None:
            return self.end_signal.value
        self.parameter.restore(params, from_cpu=True)
        state.train_count = train_count
        state.sync_actor += 1
        return self.end_signal.value


def _run_actor(
    cfg: MpConfig,
    queue_act_to_mem: mp.Queue,
    qsize_act_to_mem: sharedctypes.Synchronized,
    remote_board: Any,
    actor_id: int,
    end_signal: Any,
):
    try:
        logger.info(f"[actor{actor_id}] start.")
        context = cfg.context
        context.run_name = RunNameTypes.actor
        context.actor_id = actor_id
        context.set_device()
        env = context.env_config.make()

        # --- parameter
        parameter = context.rl_config.make_parameter(env=env)
        dat = remote_board.value
        if dat is not None:
            train_count, params = pickle.loads(dat)
            if params is not None:
                parameter.restore(params, from_cpu=True)

        # --- memory
        memory = cast(
            RLMemory,
            _ActorRLMemoryInterceptor(
                queue_act_to_mem,
                qsize_act_to_mem,
                end_signal,
                actor_id,
                context.rl_config.make_memory(env=env),
                cfg,
            ),
        )

        # --- callback
        callbacks = cfg.callbacks[:]
        callbacks.append(
            _ActorInterrupt(
                remote_board,
                parameter,
                end_signal,
                cfg,
            )
        )

        # --- play
        context.training = True
        context.disable_trainer = True
        # context.max_episodes = -1
        # context.max_memory = -1
        # context.max_steps = -1
        # context.max_train_count = -1
        # context.timeout = -1
        workers, main_worker_idx = context.rl_config.make_workers(context.players, env, parameter, memory)
        core_play.play(
            context=context,
            env=env,
            workers=workers,
            main_worker_idx=main_worker_idx,
            trainer=None,
            callbacks=cast(List[RunCallback], callbacks),
        )

    except MemoryError:
        import gc

        gc.collect()
        raise
    except Exception:
        raise
    finally:
        if not end_signal.value:
            end_signal.value = True
            logger.info(f"[actor{actor_id}] end_signal=True (actor mp end)")
        else:
            logger.info(f"[actor{actor_id}] end")


# --------------------
# memory
# --------------------
def _run_memory(
    cfg: MpConfig,
    queue_act_to_mem: mp.Queue,
    qsize_act_to_mem: sharedctypes.Synchronized,
    queue_mem_to_train: mp.Queue,
    qsize_mem_to_train_list: List[sharedctypes.Synchronized],
    queue_train_to_mem: mp.Queue,
    qsize_train_to_mem: sharedctypes.Synchronized,
    end_signal: Any,
    memory_dat,
):
    try:
        logger.info("[memory] start.")

        memory = cfg.context.rl_config.make_memory()
        memory.restore(memory_dat)
        worker_funcs = {k: v[0] for k, v in memory.get_worker_funcs().items()}
        trainer_recv_funcs = memory.get_trainer_recv_funcs()
        trainer_send_funcs = memory.get_trainer_send_funcs()

        assert len(qsize_mem_to_train_list) == len(trainer_recv_funcs)
        N = len(qsize_mem_to_train_list)
        info = {
            "memory": memory,
            "act_to_mem_size": 0,
            "act_to_mem": 0,
            "mem_to_train_size_list": [0 for _ in range(N)],
            "mem_to_train_rate_list": [20 for _ in range(N)],
            "mem_to_train_list": [0 for _ in range(N)],
            "mem_to_train_skip_list": [0 for _ in range(N)],
            "train_to_mem_size": 0,
            "train_to_mem": 0,
        }

        _calls_on_memory: List[Any] = [c for c in cfg.callbacks if hasattr(c, "on_memory")]
        [c.on_memory_start(cfg.context, info) for c in cfg.callbacks]

        while not end_signal.value:
            # --- recv worker func
            if not queue_act_to_mem.empty():
                name, raw = queue_act_to_mem.get(timeout=5)
                with qsize_act_to_mem.get_lock():
                    qsize_act_to_mem.value -= 1
                worker_funcs[name](*raw, serialized=True)
                info["act_to_mem_size"] = qsize_act_to_mem.value
                info["act_to_mem"] += 1

            # --- send trainer func
            for i in range(N):
                qsize = qsize_mem_to_train_list[i].value
                info["mem_to_train_size_list"][i] = qsize
                if qsize < cfg.mem_to_train_queue_capacity:
                    batch = trainer_recv_funcs[i]()
                    if batch is not None:
                        queue_mem_to_train.put(pickle.dumps((i, batch)))
                        with qsize_mem_to_train_list[i].get_lock():
                            qsize_mem_to_train_list[i].value += 1
                        info["mem_to_train_list"][i] += 1

            # --- recv trainer func
            if not queue_train_to_mem.empty():
                name, raw = queue_train_to_mem.get(timeout=5)
                args, kwargs = pickle.loads(raw)
                trainer_send_funcs[name](*args, **kwargs)
                with qsize_train_to_mem.get_lock():
                    qsize_train_to_mem.value -= 1
                info["train_to_mem_size"] = qsize_train_to_mem.value
                info["train_to_mem"] += 1

            [c.on_memory(cfg.context, info) for c in _calls_on_memory]

        [c.on_memory_end(cfg.context, info) for c in cfg.callbacks]

    except MemoryError:
        import gc

        gc.collect()
        raise
    except Exception:
        raise
    finally:
        if not end_signal.value:
            end_signal.value = True
            logger.info("[memory] end_signal=True (memory mp end)")
        else:
            logger.info("[memory] end")


# --------------------
# trainer
# --------------------
class _TrainerRLMemoryInterceptor:
    def __init__(
        self,
        qsize_mem_to_train_list: List[sharedctypes.Synchronized],
        queue_train_to_mem: mp.Queue,
        qsize_train_to_mem: sharedctypes.Synchronized,
        end_signal: ctypes.c_bool,
        base_memory: RLMemory,
        cfg: MpConfig,
    ):
        self.qsize_mem_to_train_list = qsize_mem_to_train_list
        self.queue_train_to_mem = queue_train_to_mem
        self.qsize_train_to_mem = qsize_train_to_mem
        self.end_signal = end_signal
        self.trainer_send_funcs = base_memory.get_trainer_send_funcs()
        self.cfg = cfg

        self.buffers = []
        self.trainer_recv_funcs = {}
        for i, func in enumerate(base_memory.get_trainer_recv_funcs()):
            self.trainer_recv_funcs[func.__name__] = i
            self.buffers.append([])

    def length(self) -> int:
        return sum([len(b) for b in self.buffers])

    def __getattr__(self, name: str) -> Callable:
        if name in self.trainer_recv_funcs:
            idx = self.trainer_recv_funcs[name]
            if len(self.buffers[idx]) == 0:
                return lambda: None
            with self.qsize_mem_to_train_list[idx].get_lock():
                self.qsize_mem_to_train_list[idx].value -= 1
            return lambda: self.buffers[idx].pop()

        if name in self.trainer_send_funcs:

            def wrapped(*args, **kwargs):
                t0 = time.time()
                while True:
                    if self.end_signal.value:
                        break

                    if self.qsize_train_to_mem.value < self.cfg.train_to_mem_queue_capacity:
                        self.queue_train_to_mem.put((name, pickle.dumps((args, kwargs))))
                        with self.qsize_train_to_mem.get_lock():
                            self.qsize_train_to_mem.value += 1
                        break

                    if time.time() - t0 > 9:
                        t0 = time.time()
                        s = f"[TrainerRLMemoryInterceptor] The queue is full so I will wait. {self.qsize_train_to_mem.value}"
                        print(s)
                        logger.info(s)
                        break
                    time.sleep(0.1)

            return wrapped

        raise AttributeError(f"{name} is not a valid method")

    def add(self, idx, batch):
        self.buffers[idx].append(batch)


def _train_parameter_communicate(
    parameter: RLParameter,
    remote_board: Any,
    end_signal: ctypes.c_bool,
    share_dict: dict,
    trainer_parameter_send_interval: int,
    exception_queue: queue.Queue,
):
    try:
        while not end_signal.value:
            time.sleep(trainer_parameter_send_interval)
            params = parameter.backup(to_cpu=True)
            if params is not None:
                remote_board.set(pickle.dumps((share_dict["train_count"], params)))
                share_dict["sync_count"] += 1
    except MemoryError:
        import gc

        gc.collect()

        logger.error(traceback.format_exc())
        end_signal.value = True
        exception_queue.put(2)
        logger.info("[trainer, parameter thread] end_signal=True (MemoryError)")
    except Exception:
        logger.error(traceback.format_exc())
        end_signal.value = True
        exception_queue.put(2)
        logger.info("[trainer, parameter thread] end_signal=True (error)")
    finally:
        logger.info("[trainer, parameter thread] end")


class _TrainerInterrupt(RunCallback):
    def __init__(
        self,
        memory: _TrainerRLMemoryInterceptor,
        queue_mem_to_train: mp.Queue,
        end_signal: ctypes.c_bool,
        share_dict: dict,
        parameter_th: threading.Thread,
        actors_ps_list: List[mp.Process],
        memory_ps: mp.Process,
    ) -> None:
        self.memory = memory
        self.queue_mem_to_train = queue_mem_to_train
        self.end_signal = end_signal
        self.share_dict = share_dict
        self.parameter_th = parameter_th
        self.actors_ps_list = actors_ps_list
        self.memory_ps = memory_ps
        self.t0_health = time.time()

    def on_train_after(self, context: RunContext, state: RunStateTrainer, **kwargs) -> bool:
        self.share_dict["train_count"] = state.trainer.train_count
        state.sync_trainer = self.share_dict["sync_count"]
        state.trainer_recv_q = self.share_dict["q_recv"]

        # --- mem -> trainer
        if not self.queue_mem_to_train.empty():
            raw = self.queue_mem_to_train.get(timeout=5)
            self.memory.add(*pickle.loads(raw))
            self.share_dict["q_recv"] += 1

        if time.time() - self.t0_health > 5:
            self.t0_health = time.time()
            if not self.parameter_th.is_alive():
                return True
            n = 0
            for w in self.actors_ps_list:
                if w.is_alive():
                    n += 1
            if n == 0:
                logger.info("all actor process dead.")
                return True
            if not self.memory_ps.is_alive():
                logger.info("memory process dead.")
                return True

        return self.end_signal.value


def _run_trainer(
    cfg: MpConfig,
    parameter: RLParameter,
    queue_mem_to_train: mp.Queue,
    qsize_mem_to_train_list: List[sharedctypes.Synchronized],
    queue_train_to_mem: mp.Queue,
    qsize_train_to_mem: sharedctypes.Synchronized,
    remote_board: Any,
    end_signal: Any,
    actors_ps_list: List[mp.Process],
    memory_ps: mp.Process,
):
    exception_queue = queue.Queue()
    try:
        logger.info("[trainer] start.")
        context = cfg.context
        context.run_name = RunNameTypes.trainer
        context.set_device()

        context.training = True
        context.distributed = True
        context.train_only = False

        share_dict = {
            "sync_count": 0,
            "q_recv": 0,
            "train_count": 0,
        }

        memory = _TrainerRLMemoryInterceptor(
            qsize_mem_to_train_list,
            queue_train_to_mem,
            qsize_train_to_mem,
            end_signal,
            context.rl_config.make_memory(),
            cfg,
        )
        trainer = context.rl_config.make_trainer(parameter, cast(RLMemory, memory))

        # --- thread
        parameter_th = threading.Thread(
            target=_train_parameter_communicate,
            args=(
                parameter,
                remote_board,
                end_signal,
                share_dict,
                cfg.trainer_parameter_send_interval,
                exception_queue,
            ),
        )
        parameter_th.start()

        # --- callback
        callbacks = cfg.callbacks[:]
        callbacks.append(
            _TrainerInterrupt(
                memory,
                queue_mem_to_train,
                end_signal,
                share_dict,
                parameter_th,
                actors_ps_list,
                memory_ps,
            )
        )

        # --- train
        core_train_only.play_trainer_only(context, trainer, callbacks)

        if not end_signal.value:
            end_signal.value = True
            logger.info("[trainer] end_signal=True (trainer end)")

        # thread end
        parameter_th.join(timeout=10)
    finally:
        # 異常終了していたら例外を出す
        if not exception_queue.empty():
            n = exception_queue.get()
            if n == 1:
                raise RuntimeError("An exception occurred in the memory thread.")
            else:
                raise RuntimeError("An exception occurred in the parameter thread.")
        logger.info("[trainer] end")


# ----------------------------
# train
# ----------------------------
__is_set_start_method = False


def train(mp_cfg: MpConfig, parameter: RLParameter, memory: RLMemory):
    global __is_set_start_method
    context = mp_cfg.context
    context.check_stop_config()

    # --- callbacks ---
    callbacks = mp_cfg.callbacks
    [c.on_start(context=context) for c in callbacks]
    # ------------------

    try:
        logger.debug(context.to_str_context())

        # mp を notebook で実行する場合はrlの定義をpyファイルにする必要あり TODO: それ以外でも動かないような
        # if is_env_notebook() and "__main__" in str(remote_memory_class):
        #    raise RuntimeError("The definition of rl must be in the py file")

        """ multiprocessing を spawn で統一

        1. tensorflowはforkに対して安全ではないようです。
        https://github.com/tensorflow/tensorflow/issues/5448#issuecomment-258934405

        2. 以下issueと多分同じバグ発生
        https://github.com/keras-team/keras/issues/3181#issuecomment-668175300

        tensorflow の内部で使っている thread が multiprocessing の子プロセスと相性が悪いようでフリーズする不具合。
        set_intra_op_parallelism_threads(1) にて子スレッドを作れないようにしても対処できる。

        set_intra_op_parallelism_threads は tensorflow の初期化前にしか実行できずエラーになるので
        こちらは実装が複雑になる可能性あり
        # RuntimeError: Intra op parallelism cannot be modified after initialization.

        3. linux + GPU + tensorflow + multiprocessing にて以下エラーがでてtrainerが落ちる
        # Failed setting context: CUDA_ERROR_NOT_INITIALIZED: initialization error
        """
        if not __is_set_start_method:
            try:
                mp.set_start_method("spawn")
            except RuntimeError:
                method = mp.get_start_method(allow_none=True)
                if method != "spawn":
                    logger.warning("Start method is not 'spawn'. Current: " + str(method))
            __is_set_start_method = True

        """ note
        - mp.Queue + BaseManager
           - qsizeはmacOSで例外が出る可能性あり(https://docs.python.org/ja/3/library/multiprocessing.html#multiprocessing.Queue)
           - 終了時にqueueにデータがあると子プロセス(actor)が終了しない
        - Manager.Queue
           - 終了時のqueue.getがものすごい時間がかかる
           - 終了時にqueueにデータがあっても終了する
           - qsizeがmacOSで使えるかは未確認
        """
        # manager.Value : https://github.com/python/cpython/issues/79967
        qsize_act_to_mem = cast(sharedctypes.Synchronized, mp.Value(ctypes.c_int, 0))
        qsize_mem_to_train_list = [
            cast(sharedctypes.Synchronized, mp.Value(ctypes.c_int, 0))
            for _ in range(len(context.rl_config.make_memory().get_trainer_recv_funcs()))  #
        ]
        qsize_train_to_mem = cast(sharedctypes.Synchronized, mp.Value(ctypes.c_int, 0))
        # manager.Valueだとなぜかうまくいかない時がある
        end_signal = mp.Value(ctypes.c_bool, False)
        with mp.Manager() as manager:
            queue_act_to_mem = mp.Queue()  # put->mp->get
            queue_mem_to_train = mp.Queue()  # put->mp->get
            queue_train_to_mem = mp.Queue()  # put->mp->get
            remote_board: Any = manager.Value(ctypes.c_char_p, None)  # set->th->mp->get

            # --- init remote_memory/parameter
            params = parameter.backup(to_cpu=True)
            if params is not None:
                remote_board.set(pickle.dumps((0, params)))

            # --- actor ---
            actors_ps_list: List[mp.Process] = []
            for actor_id in range(context.actor_num):
                params = (
                    mp_cfg,
                    queue_act_to_mem,
                    qsize_act_to_mem,
                    remote_board,
                    actor_id,
                    end_signal,
                )
                ps = mp.Process(target=_run_actor, args=params)
                actors_ps_list.append(ps)
            # -------------

            # --- memory
            memory_ps = mp.Process(
                target=_run_memory,
                args=(
                    mp_cfg,
                    queue_act_to_mem,
                    qsize_act_to_mem,
                    queue_mem_to_train,
                    qsize_mem_to_train_list,
                    queue_train_to_mem,
                    qsize_train_to_mem,
                    end_signal,
                    memory.backup(compress=True),
                ),
            )
            # --------------

            # --- start
            logger.info("[main] process start")
            [p.start() for p in actors_ps_list]
            memory_ps.start()

            # train
            try:
                _run_trainer(
                    mp_cfg,
                    parameter,
                    queue_mem_to_train,
                    qsize_mem_to_train_list,
                    queue_train_to_mem,
                    qsize_train_to_mem,
                    remote_board,
                    end_signal,
                    actors_ps_list,
                    memory_ps,
                )
            finally:
                if not end_signal.value:
                    end_signal.value = True
                    logger.info("[main] end_signal=True (trainer end)")

            # --- プロセスの終了を待つ
            for i, w in enumerate(actors_ps_list + [memory_ps]):
                for _ in range(10):
                    if w.is_alive():
                        time.sleep(1)
                    else:
                        # 子プロセスが正常終了していなければ例外を出す
                        # exitcode: 0 正常, 1 例外, 負 シグナル
                        if w.exitcode != 0 and w.exitcode is not None:
                            raise RuntimeError(f"An exception has occurred.(exitcode: {w.exitcode})")
                        break
                else:
                    logger.info("[main] terminate process.")
                    w.terminate()

    finally:
        # --- callbacks ---
        [c.on_end(context=context) for c in callbacks]
        # ------------------
