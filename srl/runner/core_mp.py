import ctypes
import logging
import multiprocessing as mp
import pprint
import time
from multiprocessing.managers import BaseManager
from typing import Any, List, Type, cast

from srl.base.rl.base import RLParameter, RLRemoteMemory
from srl.base.rl.registration import make_remote_memory
from srl.runner.callback import Callback, MPCallback, TrainerCallback
from srl.runner.runner import Config, Context, Runner

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
class Board:
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
class _ActorInterrupt(Callback):
    def __init__(
        self,
        remote_board: Board,
        parameter: RLParameter,
        train_end_signal: ctypes.c_bool,
        config: Config,
    ) -> None:
        self.remote_board = remote_board
        self.parameter = parameter
        self.train_end_signal = train_end_signal
        self.actor_parameter_sync_interval_by_step = config.actor_parameter_sync_interval_by_step

        self.step = 0
        self.prev_update_count = 0
        self.sync_count = 0

    def on_episodes_begin(self, runner: Runner):
        runner.state.sync_actor = 0

    def on_step_end(self, runner: Runner):
        self.step += 1
        if self.step % self.actor_parameter_sync_interval_by_step != 0:
            return
        update_count = self.remote_board.get_update_count()
        if update_count == self.prev_update_count:
            return
        self.prev_update_count = update_count
        params = self.remote_board.read()
        if params is None:
            return
        self.parameter.restore(params)
        runner.state.sync_actor += 1

    def intermediate_stop(self, runner: Runner) -> bool:
        if self.train_end_signal.value:
            return True
        return False


def __run_actor(
    config: Config,
    context: Context,
    remote_memory: RLRemoteMemory,
    remote_board: Board,
    actor_id: int,
    train_end_signal: ctypes.c_bool,
):
    try:
        context.run_name = f"actor{actor_id}"
        context.actor_id = actor_id
        runner = Runner(config.env_config, config.rl_config)
        runner.config = config
        runner.context = context

        # --- set_config_by_actor
        runner.config.rl_config.set_config_by_actor(
            runner.config.actor_num,
            runner.context.actor_id,
        )

        # --- parameter
        parameter = runner.make_parameter()
        params = remote_board.read()
        if params is not None:
            parameter.restore(params)

        # --- callbacks
        runner.context.callbacks.append(
            _ActorInterrupt(
                remote_board,
                parameter,
                train_end_signal,
                config,
            )
        )

        # --- train
        context.train_only = False
        context.disable_trainer = True
        context.training = True
        runner._play(parameter, remote_memory)

    finally:
        train_end_signal.value = True
        logger.info(f"actor{context.actor_id} end")


# --------------------
# trainer
# --------------------
class _TrainerInterrupt(TrainerCallback):
    def __init__(
        self,
        remote_board: Board,
        parameter: RLParameter,
        train_end_signal: ctypes.c_bool,
        config: Config,
    ) -> None:
        self.remote_board = remote_board
        self.parameter = parameter
        self.train_end_signal = train_end_signal
        self.trainer_parameter_send_interval_by_train_count = config.trainer_parameter_send_interval_by_train_count

    def on_trainer_start(self, runner: Runner):
        runner.state.sync_trainer = 0

    def on_trainer_train(self, runner: Runner):
        train_count = runner.state.trainer.get_train_count()  # type:ignore , trainer is not None

        if train_count == 0:
            time.sleep(1)
            return

        if train_count % self.trainer_parameter_send_interval_by_train_count == 0:
            self.remote_board.write(self.parameter.backup())
            runner.state.sync_trainer += 1

    def intermediate_stop(self, runner: Runner) -> bool:
        if self.train_end_signal.value:
            return True
        return False


def __run_trainer(
    config: Config,
    context: Context,
    remote_memory: RLRemoteMemory,
    remote_board: Board,
    train_end_signal: ctypes.c_bool,
):
    parameter = None
    try:
        context.run_name = "trainer"
        runner = Runner(config.env_config, config.rl_config)
        runner.config = config
        runner.context = context

        # --- parameter
        parameter = runner.make_parameter(is_load=False)
        params = remote_board.read()
        if params is not None:
            parameter.restore(params)

        # --- callbacks
        runner.context.callbacks.append(
            _TrainerInterrupt(
                remote_board,
                parameter,
                train_end_signal,
                config,
            )
        )

        # --- train
        context.train_only = True
        context.disable_trainer = False
        context.training = True
        runner._play(parameter, remote_memory)

    finally:
        train_end_signal.value = True
        if parameter is not None:
            t0 = time.time()
            remote_board.write(parameter.backup())
            logger.info(f"trainer end.(send parameter time: {time.time() - t0:.1f}s)")


# ----------------------------
# 学習
# ----------------------------
class MPManager(BaseManager):
    pass


__is_set_start_method = False


def train(
    runner: Runner,
    save_remote_memory: str,
    return_remote_memory: bool,
):
    global __is_set_start_method

    logger.info(f"Config\n{pprint.pformat(runner.config.to_json_dict())}")
    logger.info(f"Context\n{pprint.pformat(runner.context.to_json_dict())}")

    remote_memory_class = cast(
        Type[RLRemoteMemory],
        make_remote_memory(
            runner.rl_config,
            runner.make_env(),
            return_class=True,
        ),
    )

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

    MPManager.register("RemoteMemory", remote_memory_class)
    MPManager.register("Board", Board)

    logger.info("MPManager start")
    with MPManager() as manager:
        _train(runner, manager, save_remote_memory, return_remote_memory)
    logger.info("MPManager end")


def _train(
    runner: Runner,
    manager: Any,
    save_remote_memory: str,
    return_remote_memory: bool,
):
    config = runner.config
    context = runner.context

    # callbacks
    _callbacks = cast(List[MPCallback], [c for c in context.callbacks if issubclass(c.__class__, MPCallback)])
    [c.on_init(runner) for c in _callbacks]

    # --- share values
    train_end_signal = cast(ctypes.c_bool, mp.Value(ctypes.c_bool, False))
    remote_memory: RLRemoteMemory = manager.RemoteMemory(config.rl_config)
    remote_board: Board = manager.Board()

    # --- init remote_memory/parameter
    remote_memory.restore(runner.make_remote_memory().backup())
    remote_board.write(runner.make_parameter().backup())

    # --- actor
    actors_ps_list: List[mp.Process] = []
    for actor_id in range(config.actor_num):
        params = (
            config,
            context,
            remote_memory,
            remote_board,
            actor_id,
            train_end_signal,
        )
        ps = mp.Process(target=__run_actor, args=params)
        actors_ps_list.append(ps)

    # --- trainer
    if context.disable_trainer:
        trainer_ps = None
    else:
        params = (
            config,
            context,
            remote_memory,
            remote_board,
            train_end_signal,
        )
        trainer_ps = mp.Process(target=__run_trainer, args=params)

    # --- start
    logger.debug("process start")
    [p.start() for p in actors_ps_list]
    if trainer_ps is not None:
        trainer_ps.start()

    # callbacks
    [c.on_start(runner) for c in _callbacks]

    # --- wait loop
    t0 = time.time()
    while True:
        time.sleep(1)  # polling time

        # プロセスが落ちたら終了
        if trainer_ps is not None:
            if not trainer_ps.is_alive():
                train_end_signal.value = True
                logger.info("train end(trainer process dead)")
                break
        for i, w in enumerate(actors_ps_list):
            if not w.is_alive():
                train_end_signal.value = True
                logger.info(f"train end(actor {i} process dead)")
                break

        # callbacks
        [c.on_polling(runner) for c in _callbacks]

        if train_end_signal.value:
            break
    logger.info(f"loop end.(run time: {(time.time() - t0)/60:.2f}m)")

    # --- プロセスの終了を待つ
    for w in actors_ps_list:
        for _ in range(5):
            if w.is_alive():
                time.sleep(1)
            else:
                break
        else:
            w.terminate()
    if trainer_ps is not None:
        for _ in range(60 * 10):
            if trainer_ps.is_alive():
                time.sleep(1)
            else:
                break
        else:
            trainer_ps.terminate()

    # 子プロセスが正常終了していなければ例外を出す
    # exitcode: 0 正常, 1 例外, 負 シグナル
    if trainer_ps is not None and trainer_ps.exitcode != 0:
        raise RuntimeError(f"An exception has occurred in trainer process.(exitcode: {trainer_ps.exitcode})")
    for i, w in enumerate(actors_ps_list):
        if w.exitcode != 0 and w.exitcode is not None:
            raise RuntimeError(f"An exception has occurred in actor {i} process.(exitcode: {w.exitcode})")

    # --- last parameter
    t0 = time.time()
    params = remote_board.read()
    if params is not None:
        runner._parameter = None
        runner.make_parameter().restore(params)
    logger.info(f"recv parameter time: {time.time() - t0:.1f}s")

    # --- last memory
    if save_remote_memory != "":
        remote_memory.save(save_remote_memory, compress=True)
    if return_remote_memory:
        runner._remote_memory = None
        t0 = time.time()
        runner.make_remote_memory().restore(remote_memory.backup())
        logger.info(f"recv remote_memory time: {time.time() - t0:.1f}s, len {runner.remote_memory.length()}")

    # callbacks
    [c.on_end(runner) for c in _callbacks]
