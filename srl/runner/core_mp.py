import ctypes
import logging
import multiprocessing as mp
import os
import pickle
import threading
import time
import traceback
from multiprocessing.managers import BaseManager
from typing import Any, List, cast

import srl
from srl.base.define import RLMemoryTypes
from srl.base.rl.base import IRLMemoryWorker, RLMemory, RLParameter
from srl.base.run.callback import RunCallback, TrainerCallback
from srl.base.run.context import RunContext, RunNameTypes
from srl.base.run.core import RunStateActor, RunStateTrainer
from srl.runner.runner import CallbackType, TaskConfig

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
# board
# --------------------
class _Board:
    def __init__(self):
        self.params = None

    def write(self, params):
        self.params = params

    def read(self):
        return self.params


# --------------------
# actor
# --------------------
class _ActorRLMemory(IRLMemoryWorker):
    def __init__(self, remote_queue: mp.Queue, dist_queue_capacity: int, end_signal: ctypes.c_bool):
        self.remote_queue = remote_queue
        self.dist_queue_capacity = dist_queue_capacity
        self.end_signal = end_signal
        self.q_send = 0

        self.last_qsize = 0

    @property
    def memory_type(self) -> RLMemoryTypes:
        return RLMemoryTypes.NONE

    def add(self, *args) -> None:
        t0 = time.time()
        while True:
            self.last_qsize = self.remote_queue.qsize()
            if 0 <= self.last_qsize < self.dist_queue_capacity:
                self.remote_queue.put(args)
                self.q_send += 1
                break
            if self.end_signal.value:
                break
            if time.time() - t0 > 10:
                t0 = time.time()
                s = f"queue capacity over: {self.last_qsize}/{self.dist_queue_capacity}"
                print(s)
                logger.info(s)
                break  # 終了条件用に1step進める

            time.sleep(1)

    def length(self) -> int:
        return self.last_qsize


class _ActorInterrupt(RunCallback):
    def __init__(
        self,
        remote_board: _Board,
        parameter: RLParameter,
        end_signal: ctypes.c_bool,
        actor_parameter_sync_interval: int,
    ) -> None:
        self.remote_board = remote_board
        self.parameter = parameter
        self.end_signal = end_signal
        self.actor_parameter_sync_interval = actor_parameter_sync_interval

        self.t0 = time.time()
        self.t0_health = time.time()

    def on_episodes_begin(self, context: RunContext, state: RunStateActor):
        state.sync_actor = 0

    def on_step_end(self, context: RunContext, state: RunStateActor) -> bool:
        if time.time() - self.t0_health > 10:
            # getppid は重い
            if os.getppid() == 1:
                self.end_signal.value = True
                logger.info("end_signal: True")
                return True
            self.t0_health = time.time()

        if time.time() - self.t0 < self.actor_parameter_sync_interval:
            return self.end_signal.value
        state.actor_send_q = cast(_ActorRLMemory, state.memory).q_send

        self.t0 = time.time()
        params = self.remote_board.read()
        if params is None:
            return self.end_signal.value
        self.parameter.restore(pickle.loads(params), from_cpu=True)
        state.sync_actor += 1
        return self.end_signal.value


def _run_actor(
    task_config: TaskConfig,
    remote_queue: mp.Queue,
    remote_board: _Board,
    actor_id: int,
    end_signal: ctypes.c_bool,
):
    try:
        logger.info(f"actor{actor_id} start.")
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
        params = remote_board.read()
        if params is not None:
            parameter.restore(pickle.loads(params), from_cpu=True)

        # --- memory
        memory = cast(
            RLMemory,
            _ActorRLMemory(
                remote_queue,
                task_config.config.dist_queue_capacity,
                end_signal,
            ),
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
            callbacks=task_config.callbacks,
        )
    finally:
        end_signal.value = True
        logger.info("end_signal: True")
        logger.info(f"actor{actor_id} end")


# --------------------
# trainer
# --------------------
def _memory_communicate(
    memory: RLMemory,
    remote_queue: mp.Queue,
    end_signal: ctypes.c_bool,
    share_dict: dict,
):
    try:
        while not end_signal.value:
            if remote_queue.empty():
                time.sleep(0.1)
            else:
                qsize = remote_queue.qsize()
                for _ in range(qsize):
                    batch = remote_queue.get(timeout=5)
                    memory.add(*batch)
                    share_dict["q_recv"] += 1

    except Exception:
        logger.error(traceback.format_exc())
    finally:
        end_signal.value = True
        logger.info("trainer memory thread end.")


def _parameter_communicate(
    parameter: RLParameter,
    remote_board: _Board,
    end_signal: ctypes.c_bool,
    share_dict: dict,
    trainer_parameter_send_interval: int,
):
    try:
        while not end_signal.value:
            time.sleep(trainer_parameter_send_interval)
            params = parameter.backup(to_cpu=True)
            if params is not None:
                remote_board.write(pickle.dumps(params))
                share_dict["sync_count"] += 1

    except Exception:
        logger.error(traceback.format_exc())
    finally:
        end_signal.value = True
        logger.info("trainer parameter thread end.")


class _TrainerInterrupt(TrainerCallback):
    def __init__(self, end_signal: ctypes.c_bool, share_dict: dict) -> None:
        self.end_signal = end_signal
        self.share_dict = share_dict

    def on_trainer_loop(self, context: RunContext, state: RunStateTrainer) -> bool:
        if not state.is_step_trained:
            # warmupなら待機
            time.sleep(1)

        state.sync_trainer = self.share_dict["sync_count"]
        state.trainer_recv_q = self.share_dict["q_recv"]
        return self.end_signal.value


def _run_trainer(
    task_config: TaskConfig,
    parameter: RLParameter,
    memory: RLMemory,
    remote_queue: mp.Queue,
    remote_board: _Board,
    end_signal: ctypes.c_bool,
):
    logger.info("trainer start.")
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
    task_config.callbacks.append(_TrainerInterrupt(end_signal, share_dict))
    runner.context.training = True
    runner.base_run_play_trainer_only(
        parameter=parameter,
        memory=memory,
        trainer=None,
        callbacks=task_config.callbacks,
    )
    end_signal.value = True
    logger.info("end_signal: True")

    # thread end
    memory_ps.join(timeout=10)
    parameter_ps.join(timeout=10)


# ----------------------------
# train
# ----------------------------
class MPManager(BaseManager):
    pass


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

    MPManager.register("Board", _Board)

    # --- share values
    end_signal = cast(ctypes.c_bool, mp.Value(ctypes.c_bool, False))
    remote_queue = mp.Queue()
    with MPManager() as manager:
        remote_board: _Board = cast(Any, manager).Board()

        # --- init remote_memory/parameter
        parameter = runner.make_parameter()
        memory = runner.make_memory()
        params = parameter.backup(to_cpu=True)
        if params is not None:
            remote_board.write(pickle.dumps(params))

        # --- actor ---
        mp_data = runner.create_task_config(callbacks)
        actors_ps_list: List[mp.Process] = []
        for actor_id in range(runner.context.actor_num):
            params = (
                mp_data,
                remote_queue,
                remote_board,
                actor_id,
                end_signal,
            )
            ps = mp.Process(target=_run_actor, args=params)
            actors_ps_list.append(ps)
        # -------------

        # --- start
        logger.debug("process start")
        [p.start() for p in actors_ps_list]

        # train
        try:
            _run_trainer(
                mp_data,
                parameter,
                memory,
                remote_queue,
                remote_board,
                end_signal,
            )
        finally:
            end_signal.value = True
            logger.info("end_signal: True")

        # --- プロセスの終了を待つ
        for w in actors_ps_list:
            # qが残っているとactorプロセスが終わらない
            for _ in range(remote_queue.qsize()):
                remote_queue.get(timeout=1)

            for _ in range(5):
                if w.is_alive():
                    time.sleep(1)
                else:
                    break
            else:
                w.terminate()

    # 子プロセスが正常終了していなければ例外を出す
    # exitcode: 0 正常, 1 例外, 負 シグナル
    for i, w in enumerate(actors_ps_list):
        if w.exitcode != 0 and w.exitcode is not None:
            raise RuntimeError(f"An exception has occurred in actor {i} process.(exitcode: {w.exitcode})")
