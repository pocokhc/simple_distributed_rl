import ctypes
import logging
import multiprocessing as mp
import os
import pickle
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from multiprocessing import sharedctypes
from typing import Any, Callable, List, Optional, cast

from srl.base.context import RunContext
from srl.base.exception import SRLError
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.run import core_play, core_train_only
from srl.base.run.callback import RunCallback
from srl.base.run.core_play import RunStateActor
from srl.base.run.core_train_only import RunStateTrainer

# from multiprocessing.managers import ValueProxy # ValueProxy[bool] -> ctypes.c_bool

logger = logging.getLogger(__name__)


if os.environ.get("SRL_TF_GPU_INITIALIZE_DEVICES", "") == "1":
    import tensorflow as tf

    # init GPU tensorflow, 初期化は親プロセスのグローバルで実施
    # 内部で_initialize_physical_devices(初期化処理)が呼ばれる
    tf.config.list_physical_devices("GPU")
    logger.info("SRL_TF_GPU_INITIALIZE_DEVICES")


@dataclass
class MpConfig:
    context: RunContext

    polling_interval: float = 1  # sec
    queue_capacity: int = 1000
    trainer_parameter_send_interval: float = 1  # sec
    actor_parameter_sync_interval: float = 1  # sec
    train_to_mem_queue_capacity: int = 100
    mem_to_train_queue_capacity: int = 5

    # memory
    return_memory_data: bool = False
    return_memory_timeout: int = 60 * 60 * 1


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
        self.parameter.restore(params, from_serialized=True)
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
    last_worker_param_queue: mp.Queue,
):
    # import faulthandler
    # faulthandler.enable()

    try:
        logger.info(f"[actor{actor_id}] start.")
        context = cfg.context
        context.run_name = "actor"
        context.actor_id = actor_id
        context.setup_device(is_mp_main_process=False)
        env = context.env_config.make()

        # --- parameter
        parameter = context.rl_config.make_parameter()
        dat = remote_board.value
        if dat is not None:
            train_count, params = pickle.loads(dat)
            if params is not None:
                parameter.restore(params, from_serialized=True)

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
        context.callbacks.append(
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
        context.max_memory = -1
        # context.max_steps = -1
        context.max_train_count = -1
        # context.timeout = -1
        workers, main_worker_idx = context.rl_config.make_workers(context.players, env, parameter, memory)
        core_play.play(context, env, workers[main_worker_idx], workers=workers)

        if context.rl_config.use_update_parameter_from_worker():
            # actor0のみ送信
            if actor_id == 0:
                logger.info(f"[actor{actor_id}] send parameter data")
                last_worker_param_queue.put(parameter.backup(serialized=True, to_worker=True))

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
    last_mem_queue: mp.Queue,
):
    try:
        logger.info("[memory] start.")
        callbacks = cfg.context.callbacks

        memory = cfg.context.rl_config.make_memory()
        if memory_dat is not None:
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

        _calls_on_memory: List[Any] = [c for c in callbacks if hasattr(c, "on_memory")]
        [c.on_memory_start(cfg.context, info) for c in callbacks]

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

        [c.on_memory_end(cfg.context, info) for c in callbacks]
        if cfg.return_memory_data:
            logger.info("[memory] send memory data")
            last_mem_queue.put(memory.backup(compress=True))

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
            params = parameter.backup(serialized=True)
            if params is not None:
                remote_board.set(pickle.dumps((share_dict["train_count"], params)))
                share_dict["sync_count"] += 1
            time.sleep(trainer_parameter_send_interval)
    except MemoryError:
        import gc

        gc.collect()

        end_signal.value = True
        logger.error(traceback.format_exc())
        exception_queue.put(1)
        logger.error("[trainer, parameter thread] end_signal=True (MemoryError)")
    except Exception:
        end_signal.value = True
        logger.error(traceback.format_exc())
        exception_queue.put(1)
        logger.error("[trainer, parameter thread] end_signal=True (error)")
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
    ) -> None:
        self.memory = memory
        self.queue_mem_to_train = queue_mem_to_train
        self.end_signal = end_signal
        self.share_dict = share_dict
        self.parameter_th = parameter_th
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

        return self.end_signal.value


def _run_trainer(
    cfg: MpConfig,
    queue_mem_to_train: mp.Queue,
    qsize_mem_to_train_list: List[sharedctypes.Synchronized],
    queue_train_to_mem: mp.Queue,
    qsize_train_to_mem: sharedctypes.Synchronized,
    remote_board: Any,
    end_signal: Any,
):
    # import faulthandler
    # faulthandler.enable()

    exception_queue = queue.Queue()
    try:
        logger.info("[trainer] start.")
        context = cfg.context
        context.run_name = "trainer"
        context.setup_device(is_mp_main_process=False)
        context.training = True
        context.distributed = True
        context.train_only = False
        share_dict = {
            "sync_count": 0,
            "q_recv": 0,
            "train_count": 0,
        }

        parameter = context.rl_config.make_parameter()
        dat = remote_board.value
        if dat is not None:
            train_count, params = pickle.loads(dat)
            if params is not None:
                parameter.restore(params, from_serialized=True)

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
        context.callbacks.append(
            _TrainerInterrupt(
                memory,
                queue_mem_to_train,
                end_signal,
                share_dict,
                parameter_th,
            )
        )

        # --- train
        core_train_only.play_trainer_only(context, trainer)

        if not end_signal.value:
            end_signal.value = True
            logger.info("[trainer] end_signal=True (trainer end)")

        # thread end
        parameter_th.join(timeout=10)

    except MemoryError:
        import gc

        gc.collect()
        raise
    except Exception:
        raise
    finally:
        if not end_signal.value:
            end_signal.value = True
            logger.info("[trainer] end_signal=True (trainer mp end)")
        else:
            logger.info("[trainer] end")
        if not exception_queue.empty():
            raise RuntimeError("An exception occurred in the parameter thread.")


# ----------------------------
# train
# ----------------------------
__is_set_start_method = False


def train(mp_cfg: MpConfig, parameter_dat: Optional[Any] = None, memory_dat: Optional[Any] = None):
    global __is_set_start_method

    # context
    context = mp_cfg.context
    context.check_context_parameter()
    context.setup_memory_limit()
    context.setup_device(is_mp_main_process=True)

    # --- callbacks ---
    callbacks = context.callbacks
    [c.on_start(context=context) for c in callbacks]
    # ------------------

    try:
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
            last_mem_queue = mp.Queue()
            last_worker_param_queue = mp.Queue()

            # params
            remote_board.set(pickle.dumps((0, parameter_dat)))

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
                    last_worker_param_queue,
                )
                ps = mp.Process(target=_run_actor, args=params)
                actors_ps_list.append(ps)

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
                    memory_dat,
                    last_mem_queue,
                ),
            )

            # --- trainer
            trainer_ps = mp.Process(
                target=_run_trainer,
                args=(
                    mp_cfg,
                    queue_mem_to_train,
                    qsize_mem_to_train_list,
                    queue_train_to_mem,
                    qsize_train_to_mem,
                    remote_board,
                    end_signal,
                ),
            )

            # --- start
            logger.info("[main] process start")
            trainer_ps.start()
            memory_ps.start()
            [p.start() for p in actors_ps_list]
            while True:
                time.sleep(mp_cfg.polling_interval)

                if not trainer_ps.is_alive():
                    end_signal.value = True
                    logger.info("trainer process dead")
                    break

                if not memory_ps.is_alive():
                    end_signal.value = True
                    logger.info("memory process dead")
                    break

                for i, w in enumerate(actors_ps_list):
                    if not w.is_alive():
                        end_signal.value = True
                        logger.info(f"actor{i} process dead")
                        break

                if end_signal.value:
                    break

            # --- last parameter
            dat = remote_board.value
            if dat is None:
                raise SRLError("Failed to get post-training parameters.")
            _, parameter_dat = pickle.loads(dat)
            try:
                if context.rl_config.use_update_parameter_from_worker():
                    work_params_dat = last_worker_param_queue.get(timeout=60 * 30)
                    if work_params_dat is not None:
                        logger.info("actor0 parameter recved.")
                        # tf/torchがimportされる可能性あり
                        trainer_parameter = context.rl_config.make_parameter()
                        if parameter_dat is not None:
                            trainer_parameter.restore(parameter_dat, from_serialized=True)
                        worker_parameter = context.rl_config.make_parameter()
                        worker_parameter.restore(work_params_dat, from_serialized=True, from_worker=True)
                        trainer_parameter.update_from_worker_parameter(worker_parameter)
                        parameter_dat = trainer_parameter.backup(serialized=False)
            except Exception:
                logger.info(traceback.format_exc())
                logger.error("Failed to receive parameter data.")

            # --- last memory
            try:
                if mp_cfg.return_memory_data:
                    logger.info("memory data wait...")
                    t0 = time.time()
                    memory_dat = last_mem_queue.get(timeout=mp_cfg.return_memory_timeout)
                    logger.info(f"memory data recived. {time.time() - t0:.3f}s")
            except Exception:
                logger.info(traceback.format_exc())
                logger.error("Failed to receive memory data.")

            # --- プロセスの終了処理
            for i, p in enumerate(actors_ps_list + [memory_ps, trainer_ps]):
                for _ in range(10):
                    if p.is_alive():
                        time.sleep(1)
                    else:
                        # 子プロセスが正常終了していなければ例外を出す
                        # exitcode: 0 正常, 1 例外, 負 シグナル
                        if p.exitcode != 0 and p.exitcode is not None:
                            raise RuntimeError(f"An exception has occurred.(exitcode: {p.exitcode})")
                        break
                else:
                    logger.info("[main] terminate process.")
                    p.terminate()

    finally:
        # --- callbacks ---
        [c.on_end(context=context) for c in callbacks]
        # ------------------

    return parameter_dat, memory_dat
