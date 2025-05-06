import ctypes
import logging
import multiprocessing as mp
import os
import pickle
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing import sharedctypes
from typing import Any, Callable, List, Optional, cast

from srl.base.context import RunContext, RunState
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
    callbacks: List[RunCallback] = field(default_factory=list)

    polling_interval: float = 1  # sec
    queue_capacity: int = 1000
    trainer_parameter_send_interval: float = 1  # sec
    actor_parameter_sync_interval: float = 1  # sec


# --------------------
# actor
# --------------------
class _ActorRLMemoryInterceptor:
    def __init__(
        self,
        remote_queue: queue.Queue,
        remote_qsize: sharedctypes.Synchronized,
        end_signal: ctypes.c_bool,
        actor_id: int,
        base_memory: RLMemory,
        cfg: MpConfig,
    ):
        self.remote_queue = remote_queue
        self.remote_qsize = remote_qsize
        self.end_signal = end_signal
        self.actor_id = actor_id
        self.serialize_funcs = {k: v[1] for k, v in base_memory.get_worker_funcs().items()}
        self.cfg = cfg
        self.q_send = 0
        self.t0_health = time.time()  # [2]

    def length(self) -> int:
        return self.remote_qsize.value

    def __getattr__(self, name: str) -> Callable:
        if name not in self.serialize_funcs:
            raise AttributeError(f"{name} is not a valid method")
        serialize_func = self.serialize_funcs[name]

        def wrapped(*args, **kwargs):
            t0 = time.time()
            while True:
                if self.end_signal.value:
                    break

                if self.remote_qsize.value < self.cfg.queue_capacity:
                    raw = serialize_func(*args, **kwargs)
                    # 引数で展開させるためにtupleにする、なので戻り値1つのtupleのみできない
                    raw = raw if isinstance(raw, tuple) else (raw,)
                    self.remote_queue.put((name, raw))
                    with self.remote_qsize.get_lock():
                        self.remote_qsize.value += 1
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
                    s = f"[actor{self.actor_id}] The queue is full so I will wait. {self.remote_qsize.value}"
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
    remote_queue: queue.Queue,
    remote_qsize: sharedctypes.Synchronized,
    remote_board: Any,
    actor_id: int,
    end_signal: Any,
):
    try:
        logger.info(f"[actor{actor_id}] start.")
        context = cfg.context
        context.run_name = "actor"
        context.actor_id = actor_id
        context.setup_device(is_mp_main_process=False)
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
                remote_queue,
                remote_qsize,
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
        context.max_memory = -1
        # context.max_steps = -1
        context.max_train_count = -1
        # context.timeout = -1
        workers, main_worker_idx = context.rl_config.make_workers(context.players, env, parameter, memory)
        state = RunState()
        state.env = env
        state.parameter = parameter
        state.memory = memory
        state.worker = workers[main_worker_idx]
        state.workers = workers
        core_play.play(context, state, callbacks=callbacks)

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
# trainer
# --------------------
def _train_memory_communicate(
    memory: RLMemory,
    remote_queue: queue.Queue,
    remote_qsize: sharedctypes.Synchronized,
    end_signal: ctypes.c_bool,
    share_dict: dict,
    exception_queue: queue.Queue,
):
    try:
        worker_funcs = {k: v[0] for k, v in memory.get_worker_funcs().items()}

        while not end_signal.value:
            if remote_queue.empty():
                time.sleep(0.1)
            else:
                name, raw = remote_queue.get(timeout=5)
                with remote_qsize.get_lock():
                    remote_qsize.value -= 1
                worker_funcs[name](*raw, serialized=True)
                share_dict["q_recv"] += 1
    except MemoryError:
        import gc

        gc.collect()

        logger.info(traceback.format_exc())
        end_signal.value = True
        exception_queue.put(1)
        logger.error("[trainer, memory thread] end_signal=True (MemoryError)")
    except Exception:
        logger.info(traceback.format_exc())
        end_signal.value = True
        exception_queue.put(1)
        logger.error("[trainer, memory thread] end_signal=True (error)")
    finally:
        logger.info("[trainer, memory thread] end")


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
        end_signal: ctypes.c_bool,
        share_dict: dict,
        memory_th: threading.Thread,
        parameter_th: threading.Thread,
    ) -> None:
        self.end_signal = end_signal
        self.share_dict = share_dict
        self.memory_th = memory_th
        self.parameter_th = parameter_th
        self.t0_health = time.time()

    def on_train_after(self, context: RunContext, state: RunStateTrainer, **kwargs) -> bool:
        if not state.is_step_trained:
            time.sleep(1)  # warmupなら待機
        self.share_dict["train_count"] = state.trainer.train_count
        state.sync_trainer = self.share_dict["sync_count"]
        state.trainer_recv_q = self.share_dict["q_recv"]

        if time.time() - self.t0_health > 5:
            self.t0_health = time.time()
            if not self.memory_th.is_alive():
                return True
            if not self.parameter_th.is_alive():
                return True
        return self.end_signal.value


def _run_trainer(cfg: MpConfig, remote_queue: queue.Queue, remote_qsize: sharedctypes.Synchronized, remote_board: Any, end_signal: Any, memory_dat, last_mem_queue: mp.Queue):
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

        memory = context.rl_config.make_memory()

        parameter = context.rl_config.make_parameter()
        dat = remote_board.value
        if dat is not None:
            train_count, params = pickle.loads(dat)
            if params is not None:
                parameter.restore(params, from_cpu=True)

        memory_th = threading.Thread(
            target=_train_memory_communicate,
            args=(
                memory,
                remote_queue,
                remote_qsize,
                end_signal,
                share_dict,
                exception_queue,
            ),
        )
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
        memory_th.start()
        parameter_th.start()

        # --- callback
        callbacks = cfg.callbacks[:]
        callbacks.append(
            _TrainerInterrupt(
                end_signal,
                share_dict,
                memory_th,
                parameter_th,
            )
        )

        # --- train
        state = RunState()
        state.parameter = parameter
        state.memory = cast(RLMemory, memory)
        state.trainer = context.rl_config.make_trainer(parameter, memory)
        core_train_only.play_trainer_only(context, state, callbacks=callbacks)

        if not end_signal.value:
            end_signal.value = True
            logger.info("[trainer] end_signal=True (trainer end)")

        # thread end
        memory_th.join(timeout=10)
        last_mem_queue.put(memory.backup(compress=True))
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

        # 異常終了していたら例外を出す
        if not exception_queue.empty():
            n = exception_queue.get(timeout=1)
            if n == 1:
                raise RuntimeError("An exception occurred in the memory thread.")
            else:
                raise RuntimeError("An exception occurred in the parameter thread.")
        logger.info("[trainer] end")


# ----------------------------
# train
# ----------------------------
__is_set_start_method = False


def train(mp_config: MpConfig, parameter_dat: Optional[Any] = None, memory_dat: Optional[Any] = None):
    global __is_set_start_method
    context = mp_config.context
    context.check_stop_config()

    # --- callbacks ---
    callbacks = mp_config.callbacks
    [c.on_start(context=context) for c in callbacks]
    # ------------------

    try:
        logger.debug(context.to_str_context())

        # --- 実行前にrl_configのsetupを保証
        context.setup_rl_config()

        # --- deviceのセットアップ
        context.setup_device(is_mp_main_process=True)

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
        remote_qsize = cast(sharedctypes.Synchronized, mp.Value(ctypes.c_int, 0))
        # manager.Valueだとなぜかうまくいかない時がある
        end_signal = mp.Value(ctypes.c_bool, False)
        with mp.Manager() as manager:
            remote_queue = manager.Queue()
            remote_board: Any = manager.Value(ctypes.c_char_p, None)
            last_mem_queue = mp.Queue()

            # params
            remote_board.set(pickle.dumps((0, parameter_dat)))

            # --- actor ---
            actors_ps_list: List[mp.Process] = []
            for actor_id in range(context.actor_num):
                params = (
                    mp_config,
                    remote_queue,
                    remote_qsize,
                    remote_board,
                    actor_id,
                    end_signal,
                )
                ps = mp.Process(target=_run_actor, args=params)
                actors_ps_list.append(ps)

            # --- trainer
            trainer_ps = mp.Process(
                target=_run_trainer,
                args=(
                    mp_config,
                    remote_queue,
                    remote_qsize,
                    remote_board,
                    end_signal,
                    memory_dat,
                    last_mem_queue,
                ),
            )

            # --- start
            logger.info("[main] process start")
            trainer_ps.start()
            [p.start() for p in actors_ps_list]
            while True:
                time.sleep(mp_config.polling_interval)

                if not trainer_ps.is_alive():
                    end_signal.value = True
                    logger.info("trainer process dead")
                    break

                for i, w in enumerate(actors_ps_list):
                    if not w.is_alive():
                        end_signal.value = True
                        logger.info(f"actor{i} process dead")
                        break

                if end_signal.value:
                    break

            # params
            dat = remote_board.value
            if dat is not None:
                _, params_dat = pickle.loads(dat)

            # memory
            try:
                if not last_mem_queue.empty():
                    memory_dat = last_mem_queue.get(timeout=60 * 10)
            except Exception:
                logger.info(traceback.format_exc())
                logger.error("Failed to receive memory data.")

            # --- プロセスの終了を待つ
            for i, w in enumerate(actors_ps_list + [trainer_ps]):
                for _ in range(10):
                    if w.is_alive():
                        time.sleep(1)
                    else:
                        # 子プロセスが正常終了していなければ例外を出す
                        # exitcode: 0 正常, 1 例外, 負 シグナル
                        if w.exitcode != 0 and w.exitcode is not None:
                            raise RuntimeError(f"An exception has occurred in actor{i} process.(exitcode: {w.exitcode})")
                        break
                else:
                    logger.info(f"[main] actor{i} terminate")
                    w.terminate()

    finally:
        # --- callbacks ---
        [c.on_end(context=context) for c in callbacks]
        # ------------------

    return params_dat, memory_dat
