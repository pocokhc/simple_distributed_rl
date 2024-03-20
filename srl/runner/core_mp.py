import ctypes
import logging
import multiprocessing as mp
import pickle
import queue
import threading
import time
import traceback
import zlib
from multiprocessing import sharedctypes
from typing import List, Optional, cast

import srl
from srl.base.define import RLMemoryTypes
from srl.base.rl.memory import IRLMemoryWorker, RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.run.callback import RunCallback, TrainerCallback
from srl.base.run.context import RunContext, RunNameTypes
from srl.base.run.core_play import RunStateActor
from srl.base.run.core_train_only import RunStateTrainer
from srl.runner.runner import CallbackType, TaskConfig

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
    ):
        self.remote_queue = remote_queue
        self.remote_qsize = remote_qsize
        self.remote_q_capacity = remote_q_capacity
        self.end_signal = end_signal
        self.actor_id = actor_id
        self.q_send = 0
        self.t0_health = time.time()  # [2]

    @property
    def memory_type(self) -> RLMemoryTypes:
        return RLMemoryTypes.NONE

    def add(self, *args) -> None:
        t0 = time.time()
        while True:
            if self.end_signal.value:
                break
            if self.remote_qsize.value < self.remote_q_capacity:
                args = zlib.compress(pickle.dumps(args))
                self.remote_queue.put(args)
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


class _ActorInterrupt(RunCallback):
    def __init__(
        self,
        remote_board: Optional[ctypes.c_char_p],
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

    def on_episodes_begin(self, context: RunContext, state: RunStateActor):
        state.sync_actor = 0

    def on_step_end(self, context: RunContext, state: RunStateActor) -> bool:
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
        params = self.remote_board.value
        if params is None:
            return self.end_signal.value
        self.parameter.restore(pickle.loads(params), from_cpu=True)
        state.sync_actor += 1
        return self.end_signal.value


def _run_actor(
    task_config: TaskConfig,
    remote_queue: queue.Queue,
    remote_qsize: sharedctypes.Synchronized,
    remote_q_capacity: int,
    remote_board: Optional[ctypes.c_char_p],
    actor_id: int,
    end_signal: ctypes.c_bool,
):
    try:
        logger.info(f"[actor{actor_id}] start.")
        context = task_config.context
        context.run_name = RunNameTypes.actor
        context.actor_id = actor_id

        # --- runner
        runner = srl.Runner(
            context.env_config,
            context.rl_config,
            task_config.config,
            task_config.context,
        )

        # --- parameter
        parameter = runner.make_parameter(is_load=False)
        params = remote_board.value
        if params is not None:
            parameter.restore(pickle.loads(params), from_cpu=True)

        # --- memory
        memory = cast(
            RLMemory,
            _ActorRLMemory(remote_queue, remote_qsize, remote_q_capacity, end_signal, actor_id),
        )

        # --- callback
        task_config.callbacks.append(
            _ActorInterrupt(
                remote_board,
                parameter,
                end_signal,
                task_config.config.actor_parameter_sync_interval,
            )
        )

        # --- play
        runner.context.training = True
        runner.context.disable_trainer = True
        runner.base_run_play(
            parameter=parameter,
            memory=memory,
            trainer=None,
            workers=None,
            main_worker_idx=0,
            callbacks=task_config.callbacks,
            enable_generator=False,
        )
    except MemoryError:
        import gc

        gc.collect()

        logger.error(traceback.format_exc())
        logger.error(f"[actor{actor_id}] error")
    except Exception:
        logger.error(traceback.format_exc())
        logger.error(f"[actor{actor_id}] error")
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
                batch = pickle.loads(zlib.decompress(batch))
                memory.add(*batch)
                share_dict["q_recv"] += 1
    except MemoryError:
        import gc

        gc.collect()

        logger.error(traceback.format_exc())
        end_signal.value = True
        logger.info("[trainer] end_signal=True (memory thread MemoryError)")
    except Exception:
        logger.error(traceback.format_exc())
        end_signal.value = True
        logger.info("[trainer] end_signal=True (memory thread error)")
    finally:
        logger.info("[trainer] memory thread end")


def _parameter_communicate(
    parameter: RLParameter,
    remote_board: Optional[ctypes.c_char_p],
    end_signal: ctypes.c_bool,
    share_dict: dict,
    trainer_parameter_send_interval: int,
):
    try:
        while not end_signal.value:
            time.sleep(trainer_parameter_send_interval)
            params = parameter.backup(to_cpu=True)
            if params is not None:
                remote_board.set(pickle.dumps(params))
                share_dict["sync_count"] += 1
    except MemoryError:
        import gc

        gc.collect()

        logger.error(traceback.format_exc())
        end_signal.value = True
        logger.info("[trainer] end_signal=True (parameter thread MemoryError)")
    except Exception:
        logger.error(traceback.format_exc())
        end_signal.value = True
        logger.info("[trainer] end_signal=True (parameter thread error)")
    finally:
        logger.info("[trainer] parameter thread end")


class _TrainerInterrupt(TrainerCallback):
    def __init__(
        self,
        end_signal: ctypes.c_bool,
        share_dict: dict,
        memory_ps: threading.Thread,
        parameter_ps: threading.Thread,
    ) -> None:
        self.end_signal = end_signal
        self.share_dict = share_dict
        self.memory_ps = memory_ps
        self.parameter_ps = parameter_ps
        self.t0_health = time.time()

    def on_trainer_loop(self, context: RunContext, state: RunStateTrainer) -> bool:
        if not state.is_step_trained:
            time.sleep(1)  # warmupなら待機
        if time.time() - self.t0_health > 5:
            self.t0_health = time.time()
            if not self.memory_ps.is_alive():
                return True
            if not self.parameter_ps.is_alive():
                return True
        state.sync_trainer = self.share_dict["sync_count"]
        state.trainer_recv_q = self.share_dict["q_recv"]
        return self.end_signal.value


def _run_trainer(
    task_config: TaskConfig,
    parameter: RLParameter,
    memory: RLMemory,
    remote_queue: queue.Queue,
    remote_qsize: sharedctypes.Synchronized,
    remote_board: Optional[Optional[ctypes.c_char_p]],
    end_signal: ctypes.c_bool,
):
    try:
        logger.info("[trainer] start.")
        task_config.context.run_name = RunNameTypes.trainer

        # --- runner
        runner = srl.Runner(
            task_config.context.env_config,
            task_config.context.rl_config,
            task_config.config,
            task_config.context,
        )

        # --- thread
        share_dict = {
            "sync_count": 0,
            "q_recv": 0,
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
                task_config.config.trainer_parameter_send_interval,
            ),
        )
        memory_ps.start()
        parameter_ps.start()

        # --- train
        task_config.callbacks.append(
            _TrainerInterrupt(
                end_signal,
                share_dict,
                memory_ps,
                parameter_ps,
            )
        )
        runner.context.training = True
        runner.base_run_play_trainer_only(
            parameter=parameter,
            memory=memory,
            trainer=None,
            callbacks=task_config.callbacks,
        )
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


def train(runner: srl.Runner, callbacks: List[CallbackType]):
    global __is_set_start_method

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

    mp_data = runner.create_task_config(callbacks)

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
        remote_board = manager.Value(ctypes.c_char_p, None)

        # --- init remote_memory/parameter
        parameter = runner.make_parameter()
        memory = runner.make_memory()
        params = parameter.backup(to_cpu=True)
        if params is not None:
            remote_board.set(pickle.dumps(params))

        # --- actor ---
        actors_ps_list: List[mp.Process] = []
        for actor_id in range(runner.context.actor_num):
            params = (
                mp_data,
                remote_queue,
                remote_qsize,
                mp_data.config.dist_queue_capacity,
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
