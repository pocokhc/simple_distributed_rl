import ctypes
import logging
import multiprocessing as mp
import pickle
import queue
import threading
import time
import traceback
from multiprocessing.managers import BaseManager
from typing import Any, List, cast

import srl
from srl.base.rl.base import IRLMemoryTrainer, IRLMemoryWorker, RLMemory, RLParameter, RLTrainer
from srl.base.run.context import RunNameTypes
from srl.runner.callback import Callback, MPCallback, TrainerCallback
from srl.runner.runner import RunnerMPData

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
        self.update_count = 0

    def write(self, params):
        self.params = params
        self.update_count += 1

    def get_update_count(self):
        return self.update_count

    def read(self):
        return self.params


# --------------------
# actor
# --------------------
class _ActorRLMemory(IRLMemoryWorker):
    def __init__(self, memory_queue: queue.Queue):
        self.queue = memory_queue

    def add(self, *args) -> None:
        self.queue.put(args)

    def length(self) -> int:
        return -1


class _ActorInterrupt(Callback):
    def __init__(
        self,
        remote_board: _Board,
        parameter: RLParameter,
        train_end_signal: ctypes.c_bool,
        actor_parameter_sync_interval_by_step: int,
    ) -> None:
        self.remote_board = remote_board
        self.parameter = parameter
        self.train_end_signal = train_end_signal
        self.actor_parameter_sync_interval_by_step = actor_parameter_sync_interval_by_step

        self.step = 0
        self.prev_update_count = 0

    def on_episodes_begin(self, runner: srl.Runner):
        runner.state.sync_actor = 0

    def on_step_end(self, runner: srl.Runner) -> bool:
        self.step += 1
        if self.step % self.actor_parameter_sync_interval_by_step != 0:
            return self.train_end_signal.value
        update_count = self.remote_board.get_update_count()
        if update_count == self.prev_update_count:
            return self.train_end_signal.value
        self.prev_update_count = update_count
        params = self.remote_board.read()
        if params is None:
            return self.train_end_signal.value
        self.parameter.restore(params)
        runner.state.sync_actor += 1
        return self.train_end_signal.value


def _run_actor(
    mp_data: RunnerMPData,
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
        memory = cast(RLMemory, _ActorRLMemory(memory_queue))

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
                context.actor_parameter_sync_interval_by_step,
            )
        )
        runner.context.training = True
        runner.context.disable_trainer = True
        runner.core_play(trainer_only=False)

    finally:
        train_end_signal.value = True
        logger.info(f"actor{actor_id} end")


# --------------------
# trainer
# --------------------
class _RLTrainer(RLTrainer):
    def __init__(
        self,
        base_trainer: RLTrainer,
        parameter: RLParameter,
        remote_board: _Board,
        trainer_parameter_send_interval_by_train_count: int,
    ):
        super().__init__(base_trainer.config, base_trainer.parameter, base_trainer.memory)
        self.base_trainer = base_trainer
        self.parameter = parameter
        self.remote_board = remote_board
        self.trainer_parameter_send_interval_by_train_count = trainer_parameter_send_interval_by_train_count
        self.sync_count = 0
        self.count_for_sync = 0

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            time.sleep(1)
            return
        memory_sample_return = self.memory.sample(self.base_trainer.batch_size, self.train_count)
        _prev_train_count = self.base_trainer.train_count
        self.base_trainer.train_on_batchs(memory_sample_return)
        self.count_for_sync += self.base_trainer.train_count - _prev_train_count

        # --- sync parameter
        if self.count_for_sync > self.trainer_parameter_send_interval_by_train_count:
            self.count_for_sync = 0
            self.remote_board.write(self.parameter.backup())
            self.sync_count += 1

        # --- infos
        self.train_count = self.base_trainer.train_count
        self.train_info = self.base_trainer.train_info

    def train_on_batchs(self, memory_sample_return) -> None:
        raise NotImplementedError("not used")


class _TrainerInterrupt(TrainerCallback):
    def __init__(self, train_end_signal: ctypes.c_bool, trainer: _RLTrainer) -> None:
        self.train_end_signal = train_end_signal
        self.trainer = trainer

    def on_trainer_train_end(self, runner: srl.Runner) -> bool:
        runner.state.sync_trainer = self.trainer.sync_count
        return self.train_end_signal.value


class _TrainerRLMemory(IRLMemoryTrainer):
    def __init__(self, base_memory: RLMemory):
        self.base_memory = base_memory
        self.q_batch = queue.Queue()
        self.q_update = queue.Queue()
        self.trainer: RLTrainer

    def add(self, *args) -> None:
        self.base_memory.add(*args)

    def loop_update(self):
        if not self.base_memory.is_warmup_needed():
            if self.q_batch.qsize() < 5:
                self.q_batch.put(self.base_memory.sample(self.trainer.batch_size, self.trainer.train_count))
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


def _memory_mp(
    memory: _TrainerRLMemory,
    memory_queue: queue.Queue,
    exit_event: threading.Event,
    enable_prepare_batch: bool,
):
    try:
        while not exit_event.is_set():
            # --- add memory
            try:
                batch = memory_queue.get(timeout=1)
                memory.add(*batch)
            except queue.Empty:
                time.sleep(1)
            if enable_prepare_batch:
                memory.loop_update()

    except Exception:
        logger.warning(traceback.format_exc())


def _run_trainer(
    mp_data: RunnerMPData,
    parameter: RLParameter,
    memory: RLMemory,
    memory_queue: queue.Queue,
    remote_board: _Board,
    train_end_signal: ctypes.c_bool,
):
    logger.info("trainer start.")
    mp_data.context.run_name = RunNameTypes.trainer

    # --- memory
    if mp_data.context.enable_prepare_batch:
        memory = cast(RLMemory, _TrainerRLMemory(memory))

    # --- runner
    runner = srl.Runner(
        mp_data.context.env_config,
        mp_data.context.rl_config,
        mp_data.config,
        mp_data.context,
        parameter,
        memory,
    )

    # --- trainer
    trainer = _RLTrainer(
        runner.make_trainer(),
        parameter,
        remote_board,
        mp_data.context.trainer_parameter_send_interval_by_train_count,
    )
    if mp_data.context.enable_prepare_batch:
        memory.trainer = trainer  # type: ignore

    # --- memory
    exit_event = threading.Event()
    memory_ps = threading.Thread(
        target=_memory_mp, args=(memory, memory_queue, exit_event, mp_data.context.enable_prepare_batch)
    )
    memory_ps.start()

    # --- train
    runner.context.callbacks.insert(0, _TrainerInterrupt(train_end_signal, trainer))
    runner.context.training = True
    runner.core_play(trainer=trainer, trainer_only=True)
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
    mp_data = runner.create_mp_data()

    # callbacks
    _callbacks = cast(List[MPCallback], [c for c in mp_data.context.callbacks if issubclass(c.__class__, MPCallback)])
    [c.on_init(runner) for c in _callbacks]

    # --- share values
    train_end_signal = cast(ctypes.c_bool, mp.Value(ctypes.c_bool, False))
    memory_queue: queue.Queue = manager.Queue()
    remote_board: _Board = manager.Board()

    # --- init remote_memory/parameter
    parameter = runner.make_parameter()
    memory = runner.make_memory()
    remote_board.write(parameter.backup())

    # --- actor
    actors_ps_list: List[mp.Process] = []
    for actor_id in range(mp_data.context.actor_num):
        params = (
            mp_data,
            memory_queue,
            remote_board,
            actor_id,
            train_end_signal,
        )
        ps = mp.Process(target=_run_actor, args=params)
        actors_ps_list.append(ps)

    # --- start
    logger.debug("process start")
    [p.start() for p in actors_ps_list]

    # callbacks
    [c.on_start(runner) for c in _callbacks]

    # train
    _run_trainer(
        mp_data,
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
