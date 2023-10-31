import ctypes
import logging
import multiprocessing as mp
import queue
import threading
import time
import traceback
from multiprocessing.managers import BaseManager
from typing import Any, List, cast

import srl
from srl.base.rl.base import IRLMemoryTrainer, IRLMemoryWorker, RLMemory, RLParameter
from srl.base.run.context import RunNameTypes
from srl.runner.callback import Callback, MPCallback, TrainerCallback
from srl.runner.runner import TaskConfig

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
    def __init__(self, memory_queue: queue.Queue, dist_queue_capacity: int, train_end_signal: ctypes.c_bool):
        self.queue = memory_queue
        self.dist_queue_capacity = dist_queue_capacity
        self.train_end_signal = train_end_signal

    def add(self, *args) -> None:
        t0 = time.time()
        while True:
            qsize = self.queue.qsize()
            if 0 <= qsize < self.dist_queue_capacity:
                self.queue.put(args)
                break
            if self.train_end_signal.value:
                break
            if time.time() - t0 > 10:
                t0 = time.time()
                print(f"capacity over, wait queue: {qsize}/{self.dist_queue_capacity}")
            time.sleep(1)

    def length(self) -> int:
        return self.queue.qsize()


class _ActorInterrupt(Callback):
    def __init__(
        self,
        remote_board: _Board,
        parameter: RLParameter,
        train_end_signal: ctypes.c_bool,
        actor_parameter_sync_interval: int,
    ) -> None:
        self.remote_board = remote_board
        self.parameter = parameter
        self.train_end_signal = train_end_signal
        self.actor_parameter_sync_interval = actor_parameter_sync_interval

        self.t0 = time.time()

    def on_episodes_begin(self, runner: srl.Runner):
        runner.state.sync_actor = 0

    def on_step_end(self, runner: srl.Runner) -> bool:
        if time.time() - self.t0 < self.actor_parameter_sync_interval:
            return self.train_end_signal.value
        self.t0 = time.time()
        params = self.remote_board.read()
        if params is None:
            return self.train_end_signal.value
        self.parameter.restore(params)
        runner.state.sync_actor += 1
        return self.train_end_signal.value


def _run_actor(
    mp_data: TaskConfig,
    memory_queue: queue.Queue,
    remote_board: _Board,
    actor_id: int,
    train_end_signal: ctypes.c_bool,
):
    try:
        logger.info(f"actor{actor_id} start.")
        context = mp_data.context
        context.run_name = RunNameTypes.actor
        context.actor_id = actor_id

        # --- set_config_by_actor
        context.rl_config.set_config_by_actor(context.actor_num, context.actor_id)

        # --- memory
        memory = cast(RLMemory, _ActorRLMemory(memory_queue, mp_data.config.dist_queue_capacity, train_end_signal))

        # --- runner
        runner = srl.Runner(
            context.env_config,
            context.rl_config,
            mp_data.config,
            context,
            memory=memory,
        )

        # --- parameter
        parameter = runner.make_parameter(is_load=False)
        params = remote_board.read()
        if params is not None:
            parameter.restore(params)

        # --- callback
        runner.context.callbacks.append(
            _ActorInterrupt(
                remote_board,
                parameter,
                train_end_signal,
                mp_data.config.actor_parameter_sync_interval,
            )
        )
        runner.context.training = True
        runner.context.disable_trainer = True
        runner.core_play(
            trainer_only=False,
            parameter=parameter,
            memory=memory,
            trainer=None,
            workers=None,
        )

    finally:
        train_end_signal.value = True
        logger.info(f"actor{actor_id} end")


# --------------------
# trainer
# --------------------
class _TrainerRLMemoryPrepareBatch(IRLMemoryTrainer):
    def __init__(self, base_memory: RLMemory, batch_size: int, share_dict: dict):
        # recv,warmup両方threadなので、両方待つ場合は待機
        self.base_memory = base_memory
        self.batch_size = batch_size
        self.share_dict = share_dict
        self.q_batch = queue.Queue()
        self.q_update = queue.Queue()

    def recv(self, dat) -> None:
        if dat is not None:
            self.base_memory.add(*dat)
        if dat is None and self.base_memory.is_warmup_needed():
            time.sleep(1)
        if not self.base_memory.is_warmup_needed():
            if self.q_batch.qsize() < 5:
                self.q_batch.put(self.base_memory.sample(self.batch_size, self.share_dict["train_count"]))
        if not self.q_update.empty():
            self.base_memory.update(self.q_update.get())

    def length(self) -> int:
        return self.base_memory.length()

    def is_warmup_needed(self) -> bool:
        return self.q_batch.empty()

    def sample(self, batch_size: int, step: int) -> Any:
        return self.q_batch.get()

    def update(self, memory_update_args: Any) -> None:
        self.q_update.put(memory_update_args)


class _TrainerRLMemory(IRLMemoryTrainer):
    def __init__(self, base_memory: RLMemory):
        # thread(recv) は受信できなければ待機
        # main(is_warmup_needed) はwarmup中なら待機
        self.base_memory = base_memory

    def recv(self, dat) -> None:
        if dat is None:
            time.sleep(1)
        else:
            self.base_memory.add(*dat)

    def length(self) -> int:
        return self.base_memory.length()

    def is_warmup_needed(self) -> bool:
        if self.base_memory.is_warmup_needed():
            time.sleep(1)
            return True
        return False

    def sample(self, batch_size: int, step: int) -> Any:
        return self.base_memory.sample(batch_size, step)

    def update(self, memory_update_args: Any) -> None:
        self.base_memory.update(memory_update_args)


def _server_communicate(
    memory: _TrainerRLMemory,
    memory_queue: queue.Queue,
    parameter: RLParameter,
    remote_board: _Board,
    exit_event: threading.Event,
    share_dict: dict,
    trainer_parameter_send_interval: int,
):
    try:
        t0 = time.time()
        while not exit_event.is_set():
            # --- memory
            try:
                if memory_queue.empty():
                    batch = None
                else:
                    batch = memory_queue.get(timeout=1)
            except queue.Empty:
                batch = None
            memory.recv(batch)

            # --- parameter
            if time.time() - t0 > trainer_parameter_send_interval:
                t0 = time.time()

                remote_board.write(parameter.backup())
                share_dict["sync_count"] += 1

    except Exception:
        logger.warning(traceback.format_exc())


class _TrainerInterrupt(TrainerCallback):
    def __init__(self, train_end_signal: ctypes.c_bool, share_dict: dict) -> None:
        self.train_end_signal = train_end_signal
        self.share_dict = share_dict

    def on_trainer_loop(self, runner: srl.Runner) -> bool:
        assert runner.state.trainer is not None
        self.share_dict["train_count"] = runner.state.trainer.get_train_count()
        runner.state.sync_trainer = self.share_dict["sync_count"]
        return self.train_end_signal.value


def _run_trainer(
    task_config: TaskConfig,
    parameter: RLParameter,
    memory: RLMemory,
    memory_queue: queue.Queue,
    remote_board: _Board,
    train_end_signal: ctypes.c_bool,
):
    logger.info("trainer start.")
    task_config.context.run_name = RunNameTypes.trainer

    # --- runner
    runner = srl.Runner(
        task_config.context.env_config,
        task_config.context.rl_config,
        task_config.config,
        task_config.context,
        parameter,
    )

    # --- memory_ps
    share_dict = {"sync_count": 0, "train_count": 0}
    if task_config.config.dist_enable_prepare_sample_batch:
        batch_size = getattr(task_config.context.rl_config, "batch_size", 1)
        memory = cast(RLMemory, _TrainerRLMemoryPrepareBatch(memory, batch_size, share_dict))
    else:
        memory = cast(RLMemory, _TrainerRLMemory(memory))
    exit_event = threading.Event()
    memory_ps = threading.Thread(
        target=_server_communicate,
        args=(
            memory,
            memory_queue,
            parameter,
            remote_board,
            exit_event,
            share_dict,
            task_config.config.trainer_parameter_send_interval,
        ),
    )
    memory_ps.start()

    # --- train
    runner.context.callbacks.append(_TrainerInterrupt(train_end_signal, share_dict))
    runner.context.training = True
    runner.core_play(
        trainer_only=True,
        parameter=parameter,
        memory=memory,
        trainer=None,
        workers=None,
    )
    train_end_signal.value = True

    # thread end
    exit_event.set()
    memory_ps.join(timeout=10)


# ----------------------------
# 学習
# ----------------------------
class MPManager(BaseManager):
    pass


__is_set_start_method = False


def train(runner: srl.Runner):
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

    MPManager.register("Queue", queue.Queue)
    MPManager.register("Board", _Board)

    logger.info("MPManager start")
    with MPManager() as manager:
        _train(runner, manager)
    logger.info("MPManager end")


def _train(runner: srl.Runner, manager: Any):
    # callbacks
    _callbacks = cast(List[MPCallback], [c for c in runner.context.callbacks if issubclass(c.__class__, MPCallback)])
    [c.on_init(runner) for c in _callbacks]

    # --- share values
    train_end_signal = cast(ctypes.c_bool, mp.Value(ctypes.c_bool, False))
    memory_queue: queue.Queue = manager.Queue()
    remote_board: _Board = manager.Board()

    # --- init remote_memory/parameter
    parameter = runner.make_parameter()
    memory = runner.make_memory()
    remote_board.write(parameter.backup())

    # --- actor ---
    mp_data = runner.create_task_config(exclude_callbacks=["Checkpoint"])
    actors_ps_list: List[mp.Process] = []
    for actor_id in range(runner.context.actor_num):
        params = (
            mp_data,
            memory_queue,
            remote_board,
            actor_id,
            train_end_signal,
        )
        ps = mp.Process(target=_run_actor, args=params)
        actors_ps_list.append(ps)
    # -------------

    # --- start
    logger.debug("process start")
    [p.start() for p in actors_ps_list]

    # callbacks
    [c.on_start(runner) for c in _callbacks]

    # train
    _run_trainer(
        runner.create_task_config(),
        parameter,
        memory,
        memory_queue,
        remote_board,
        train_end_signal,
    )

    # --- プロセスの終了を待つ
    for w in actors_ps_list:
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

    # callbacks
    [c.on_end(runner) for c in _callbacks]
