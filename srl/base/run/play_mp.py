import ctypes
import logging
import multiprocessing as mp
import pickle
import pprint
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing import sharedctypes
from typing import Any, List, cast

from srl.base.context import RunContext, RunNameTypes
from srl.base.rl.memory import IRLMemoryWorker, RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.run import core_play, core_train_only
from srl.base.run.callback import CallbackType, RunCallback, TrainCallback
from srl.base.run.core_play import RunStateActor
from srl.base.run.core_train_only import RunStateTrainer

# from multiprocessing.managers import ValueProxy # ValueProxy[bool] -> ctypes.c_bool

logger = logging.getLogger(__name__)

"""
RuntimeError:
    An attempt has been made to start a new process before the
    current process has finished its bootstrapping phase.
    This probably means that you are not using fork to start your
    child processes and you have forgotten to use the proper idiom
    in the main module:

        if __name__ == '__main__':
            freeze_support()
            ...

    The "freeze_support()" line can be omitted if the program
    is not going to be frozen to produce an executable.

"""


@dataclass
class MpData:
    context: RunContext
    callbacks: List[CallbackType] = field(default_factory=list)

    queue_capacity: int = 1000
    trainer_parameter_send_interval: int = 1  # sec
    actor_parameter_sync_interval: int = 1  # sec


# --------------------
# actor
# --------------------
class _ActorRLMemory(IRLMemoryWorker):
    def __init__(
        self,
        remote_queue: queue.Queue,
        remote_qsize: sharedctypes.Synchronized,
        remote_q_capacity: int,
        end_signal: ctypes.c_bool,
        actor_id: int,
        base_memory: IRLMemoryWorker,
    ):
        self.remote_queue = remote_queue
        self.remote_qsize = remote_qsize
        self.remote_q_capacity = remote_q_capacity
        self.end_signal = end_signal
        self.actor_id = actor_id
        self.base_memory = base_memory
        self.q_send = 0
        self.t0_health = time.time()  # [2]

    def add(self, *args) -> None:
        t0 = time.time()
        while True:
            if self.end_signal.value:
                break
            if self.remote_qsize.value < self.remote_q_capacity:
                raw = self.base_memory.serialize_add_args(*args)
                self.remote_queue.put(raw)
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

    def length(self) -> int:
        return self.remote_qsize.value

    def serialize_add_args(self, *args):
        raise NotImplementedError("Unused")


class _ActorInterrupt(RunCallback):
    def __init__(
        self,
        remote_board: Any,
        parameter: RLParameter,
        end_signal: ctypes.c_bool,
        actor_parameter_sync_interval: int,
    ) -> None:
        self.remote_board = remote_board
        self.parameter = parameter
        self.end_signal = end_signal
        self.actor_parameter_sync_interval = actor_parameter_sync_interval

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

        if time.time() - self.t0 < self.actor_parameter_sync_interval:
            return self.end_signal.value
        state.actor_send_q = cast(_ActorRLMemory, state.memory).q_send

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
    mp_data: MpData,
    remote_queue: queue.Queue,
    remote_qsize: sharedctypes.Synchronized,
    remote_q_capacity: int,
    remote_board: Any,
    actor_id: int,
    end_signal: Any,
):
    try:
        logger.info(f"[actor{actor_id}] start.")
        context = mp_data.context
        context.run_name = RunNameTypes.actor
        context.actor_id = actor_id
        context.set_device()
        env = context.env_config.make()

        # --- parameter
        parameter = context.rl_config.make_parameter(env=env, is_load=False)
        dat = remote_board.value
        if dat is not None:
            train_count, params = pickle.loads(dat)
            if params is not None:
                parameter.restore(params, from_cpu=True)

        # --- memory
        memory = _ActorRLMemory(
            remote_queue,
            remote_qsize,
            remote_q_capacity,
            end_signal,
            actor_id,
            context.rl_config.make_memory(env=env, is_load=False),
        )

        # --- callback
        callbacks = [c for c in mp_data.callbacks if issubclass(c.__class__, RunCallback)]
        callbacks.append(
            _ActorInterrupt(
                remote_board,
                parameter,
                end_signal,
                mp_data.actor_parameter_sync_interval,
            )
        )

        # --- play
        context.training = True
        context.disable_trainer = True
        context.enable_train_thread = False
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
            logger.info(f"[actor{actor_id}] end_signal=True (actor thread end)")
        else:
            logger.info(f"[actor{actor_id}] end")


# --------------------
# trainer
# --------------------
def _memory_communicate(
    memory: RLMemory,
    remote_queue: queue.Queue,
    remote_qsize: sharedctypes.Synchronized,
    end_signal: ctypes.c_bool,
    share_dict: dict,
):
    try:
        while not end_signal.value:
            if remote_queue.empty():
                time.sleep(0.1)
            else:
                batch = remote_queue.get(timeout=5)
                with remote_qsize.get_lock():
                    remote_qsize.value -= 1
                batch = memory.deserialize_add_args(batch)
                memory.add(*batch)
                share_dict["q_recv"] += 1
    except MemoryError:
        import gc

        gc.collect()

        logger.error(traceback.format_exc())
        end_signal.value = True
        logger.info("[trainer, memory thread] end_signal=True (MemoryError)")
    except Exception:
        logger.error(traceback.format_exc())
        end_signal.value = True
        logger.info("[trainer, memory thread] end_signal=True (error)")
    finally:
        logger.info("[trainer, memory thread] end")


def _parameter_communicate(
    parameter: RLParameter,
    remote_board: Any,
    end_signal: ctypes.c_bool,
    share_dict: dict,
    trainer_parameter_send_interval: int,
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
        logger.info("[trainer, parameter thread] end_signal=True (MemoryError)")
    except Exception:
        logger.error(traceback.format_exc())
        end_signal.value = True
        logger.info("[trainer, parameter thread] end_signal=True (error)")
    finally:
        logger.info("[trainer, parameter thread] end")


class _TrainerInterrupt(TrainCallback):
    def __init__(
        self,
        end_signal: ctypes.c_bool,
        share_dict: dict,
        memory_ps: threading.Thread,
        parameter_ps: threading.Thread,
        actors_ps_list: List[mp.Process],
    ) -> None:
        self.end_signal = end_signal
        self.share_dict = share_dict
        self.memory_ps = memory_ps
        self.parameter_ps = parameter_ps
        self.actors_ps_list = actors_ps_list
        self.t0_health = time.time()

    def on_train_after(self, context: RunContext, state: RunStateTrainer, **kwargs) -> bool:
        if not state.is_step_trained:
            time.sleep(1)  # warmupなら待機
        self.share_dict["train_count"] = state.trainer.train_count
        state.sync_trainer = self.share_dict["sync_count"]
        state.trainer_recv_q = self.share_dict["q_recv"]
        if time.time() - self.t0_health > 5:
            self.t0_health = time.time()
            if not self.memory_ps.is_alive():
                return True
            if not self.parameter_ps.is_alive():
                return True
            n = 0
            for w in self.actors_ps_list:
                if w.is_alive():
                    n += 1
            if n == 0:
                logger.info("all actor process dead.")
                return True
        return self.end_signal.value


def _run_trainer(
    mp_data: MpData,
    parameter: RLParameter,
    memory: RLMemory,
    remote_queue: queue.Queue,
    remote_qsize: sharedctypes.Synchronized,
    remote_board: Any,
    end_signal: Any,
    actors_ps_list: List[mp.Process],
):
    try:
        logger.info("[trainer] start.")
        context = mp_data.context
        context.run_name = RunNameTypes.trainer
        context.set_device()

        # --- thread
        share_dict = {
            "sync_count": 0,
            "q_recv": 0,
            "train_count": 0,
        }
        memory_ps = threading.Thread(
            target=_memory_communicate,
            args=(
                memory,
                remote_queue,
                remote_qsize,
                end_signal,
                share_dict,
            ),
        )
        parameter_ps = threading.Thread(
            target=_parameter_communicate,
            args=(
                parameter,
                remote_board,
                end_signal,
                share_dict,
                mp_data.trainer_parameter_send_interval,
            ),
        )
        memory_ps.start()
        parameter_ps.start()

        # --- callback
        callbacks = [c for c in mp_data.callbacks if issubclass(c.__class__, TrainCallback)]
        callbacks.append(
            _TrainerInterrupt(
                end_signal,
                share_dict,
                memory_ps,
                parameter_ps,
                actors_ps_list,
            )
        )

        # --- train
        context.training = True
        context.distributed = True
        context.train_only = False
        trainer = context.rl_config.make_trainer(parameter, memory)
        core_train_only.play_trainer_only(context, trainer, cast(List[TrainCallback], callbacks))

        if not end_signal.value:
            end_signal.value = True
            logger.info("[trainer] end_signal=True (trainer end)")

        # thread end
        memory_ps.join(timeout=10)
        parameter_ps.join(timeout=10)
    finally:
        logger.info("[trainer] end")


# ----------------------------
# train
# ----------------------------
__is_set_start_method = False


def train(
    mp_data: MpData,
    parameter: RLParameter,
    memory: RLMemory,
    logger_config: bool = False,
):
    global __is_set_start_method
    context = mp_data.context
    context.check_stop_config()

    # --- callbacks ---
    callbacks = mp_data.callbacks
    [c.on_start(context=context) for c in callbacks]
    # ------------------

    try:
        # --- log ---
        if logger_config:
            logger.info("--- EnvConfig ---" + "\n" + pprint.pformat(context.env_config.to_dict()))
            logger.info("--- RLConfig ---" + "\n" + pprint.pformat(context.rl_config.to_dict()))
            logger.info("--- Context ---" + "\n" + pprint.pformat(context.to_dict(include_env_config=False, include_rl_config=False)))
        # ------------

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
            if mp.get_start_method() != "spawn":
                mp.set_start_method("spawn", force=True)
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

            # --- init remote_memory/parameter
            params = parameter.backup(to_cpu=True)
            if params is not None:
                remote_board.set(pickle.dumps((0, params)))

            # --- actor ---
            actors_ps_list: List[mp.Process] = []
            for actor_id in range(context.actor_num):
                params = (
                    mp_data,
                    remote_queue,
                    remote_qsize,
                    mp_data.queue_capacity,
                    remote_board,
                    actor_id,
                    end_signal,
                )
                ps = mp.Process(target=_run_actor, args=params)
                actors_ps_list.append(ps)
            # -------------

            # --- start
            logger.info("[main] process start")
            [p.start() for p in actors_ps_list]

            # train
            try:
                _run_trainer(
                    mp_data,
                    parameter,
                    memory,
                    remote_queue,
                    remote_qsize,
                    remote_board,
                    end_signal,
                    actors_ps_list,
                )
            finally:
                if not end_signal.value:
                    end_signal.value = True
                    logger.info("[main] end_signal=True (trainer end)")

            # --- プロセスの終了を待つ
            for i, w in enumerate(actors_ps_list):
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
