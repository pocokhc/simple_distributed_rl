import ctypes
import logging
import multiprocessing as mp
import os
import pprint
import time
import traceback
from multiprocessing.managers import BaseManager
from typing import Any, List, Optional, Type, Union, cast

from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory
from srl.base.rl.registration import make_remote_memory
from srl.runner import play_sequence
from srl.runner.callback import Callback
from srl.runner.callbacks.file_log_reader import FileLogReader
from srl.runner.callbacks.print_progress import PrintProgress
from srl.runner.config import Config
from srl.runner.play_trainer import play as train_only
from srl.utils.common import is_enable_tf_device_name

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

    def on_episodes_begin(self, info):
        info["sync"] = 0

    def on_step_end(self, info):
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
        self.sync_count += 1
        info["sync"] = self.sync_count

    def intermediate_stop(self, info) -> bool:
        if self.train_end_signal.value:
            return True
        return False


def __run_actor(
    config: Config,
    remote_memory: RLRemoteMemory,
    remote_board: Board,
    actor_id: int,
    train_end_signal: ctypes.c_bool,
):
    config.run_name = f"actor{actor_id}"
    config.run_actor_id = actor_id
    config.init_process()

    allocate = config.used_device_tf
    if (not config.tf_disable) and is_enable_tf_device_name(allocate):
        import tensorflow as tf

        with tf.device(allocate):
            logger.info(f"actor{actor_id} start(allocate={allocate})")
            __run_actor_main(config, remote_memory, remote_board, actor_id, train_end_signal)
    else:
        logger.info(f"actor{actor_id} start.")
        __run_actor_main(config, remote_memory, remote_board, actor_id, train_end_signal)


def __run_actor_main(
    config: Config,
    remote_memory: RLRemoteMemory,
    remote_board: Board,
    actor_id: int,
    train_end_signal: ctypes.c_bool,
):
    try:
        # --- config
        config.disable_trainer = True
        config.rl_config.set_config_by_actor(config.actor_num, actor_id)

        # --- parameter
        parameter = config.make_parameter(is_load=False)
        params = remote_board.read()
        if params is not None:
            parameter.restore(params)

        # --- callbacks
        config.callbacks.append(
            _ActorInterrupt(
                remote_board,
                parameter,
                train_end_signal,
                config,
            )
        )

        # --- play
        play_sequence.play(config, parameter, remote_memory, actor_id)

    finally:
        train_end_signal.value = True
        logger.info(f"actor{actor_id} end")


# --------------------
# trainer
# --------------------
class _TrainerInterrupt(Callback):
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
        self.config = config

        self.sync_count = 0

    def on_trainer_start(self, info):
        info["sync"] = 0

    def on_trainer_train(self, info):
        train_count = info["train_count"]

        if train_count == 0:
            time.sleep(1)
            return

        if train_count % self.config.trainer_parameter_send_interval_by_train_count == 0:
            self.remote_board.write(self.parameter.backup())
            self.sync_count += 1
            info["sync"] = self.sync_count

    def intermediate_stop(self, info) -> bool:
        if self.train_end_signal.value:
            return True
        return False


def __run_trainer(
    config: Config,
    remote_memory: RLRemoteMemory,
    remote_board: Board,
    train_end_signal: ctypes.c_bool,
):
    config.run_name = "trainer"
    config.init_process()

    allocate = config.used_device_tf
    if (not config.tf_disable) and is_enable_tf_device_name(allocate):
        import tensorflow as tf

        with tf.device(allocate):
            logger.info(f"trainer start(allocate={allocate})")
            __run_trainer_main(config, remote_memory, remote_board, train_end_signal)
    else:
        logger.info("trainer start.")
        __run_trainer_main(config, remote_memory, remote_board, train_end_signal)


def __run_trainer_main(
    config: Config,
    remote_memory: RLRemoteMemory,
    remote_board: Board,
    train_end_signal: ctypes.c_bool,
):
    # --- parameter
    parameter = config.make_parameter(is_load=False)
    params = remote_board.read()
    if params is not None:
        parameter.restore(params)

    try:
        # --- callbacks
        config.callbacks.append(
            _TrainerInterrupt(
                remote_board,
                parameter,
                train_end_signal,
                config,
            )
        )

        # --- play
        train_only(config, parameter, remote_memory)

    finally:
        train_end_signal.value = True
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
    config: Config,
    # stop config
    max_episodes: int = -1,
    timeout: int = -1,
    max_steps: int = -1,
    max_train_count: int = -1,
    # play config
    shuffle_player: bool = True,
    disable_trainer: bool = False,
    enable_profiling: bool = True,
    # evaluate
    enable_evaluation: bool = True,
    eval_env_sharing: bool = True,
    eval_interval: int = 0,  # episode
    eval_num_episode: int = 1,
    eval_players: List[Union[None, str, RLConfig]] = [],
    # PrintProgress
    print_progress: bool = True,
    progress_max_time: int = 60 * 10,  # s
    progress_start_time: int = 5,
    progress_print_env_info: bool = False,
    progress_print_worker_info: bool = True,
    progress_print_train_info: bool = True,
    progress_print_worker: int = 0,
    progress_max_actor: int = 5,
    # file_log
    enable_file_logger: bool = True,
    file_logger_tmp_dir: str = "tmp",
    file_logger_enable_train_log: bool = True,
    file_logger_train_log_interval: int = 1,  # s
    file_logger_enable_checkpoint: bool = True,
    file_logger_checkpoint_interval: int = 60 * 20,  # s
    # other
    callbacks: List[Callback] = [],
    init_parameter: Optional[RLParameter] = None,
    init_remote_memory: Optional[RLRemoteMemory] = None,
    return_memory: bool = False,
    save_memory: str = "",
):
    global __is_set_start_method

    if disable_trainer:
        enable_evaluation = False  # 学習しないので
        assert (
            max_steps != -1 or max_episodes != -1 or timeout != -1
        ), "Please specify 'max_episodes', 'timeout' or 'max_steps'."
    else:
        assert (
            max_steps != -1 or max_episodes != -1 or timeout != -1 or max_train_count != -1
        ), "Please specify 'max_episodes', 'timeout' , 'max_steps' or 'max_train_count'."

    config = config.copy(env_share=False)

    # stop config
    config.max_episodes = max_episodes
    config.timeout = timeout
    config.max_steps = max_steps
    config.max_train_count = max_train_count
    # play config
    config.shuffle_player = shuffle_player
    config.disable_trainer = disable_trainer
    config.enable_profiling = enable_profiling
    # callbacks
    config.callbacks = callbacks[:]
    # play info
    config.training = True
    config.distributed = True

    # --- Evaluate(最初に追加)
    if enable_evaluation:
        from srl.runner.callbacks.evaluate import Evaluate

        config.callbacks.insert(
            0,
            Evaluate(
                env_sharing=eval_env_sharing,
                interval=eval_interval,
                num_episode=eval_num_episode,
                eval_players=eval_players,
            ),
        )

    # --- PrintProgress
    if print_progress:
        config.callbacks.append(
            PrintProgress(
                max_time=progress_max_time,
                start_time=progress_start_time,
                print_env_info=progress_print_env_info,
                print_worker_info=progress_print_worker_info,
                print_train_info=progress_print_train_info,
                print_worker=progress_print_worker,
                max_actor=progress_max_actor,
            )
        )

    # --- FileLog
    if enable_file_logger:
        from srl.runner.callbacks.file_log_writer import FileLogWriter

        file_logger = FileLogWriter(
            tmp_dir=file_logger_tmp_dir,
            enable_train_log=file_logger_enable_train_log,
            train_log_interval=file_logger_train_log_interval,
            enable_episode_log=False,
            enable_checkpoint=file_logger_enable_checkpoint,
            checkpoint_interval=file_logger_checkpoint_interval,
        )
        config.callbacks.append(file_logger)
    else:
        file_logger = None

    # ---------------------------------------------
    logger.info(f"Training Config\n{pprint.pformat(config.to_dict())}")
    config.assert_params()

    remote_memory_class = cast(Type[RLRemoteMemory], make_remote_memory(config.rl_config, return_class=True))

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
        mp.set_start_method("spawn")
        __is_set_start_method = True

    MPManager.register("RemoteMemory", remote_memory_class)
    MPManager.register("Board", Board)

    logger.info("MPManager start")
    with MPManager() as manager:
        return_parameter, return_remote_memory = _train(
            config,
            init_parameter,
            init_remote_memory,
            manager,
            disable_trainer,
            return_memory,
            save_memory,
        )
    logger.info("MPManager end")

    # --- history
    history = FileLogReader()
    try:
        if file_logger is not None:
            history.load(file_logger.base_dir)
    except Exception:
        logger.info(traceback.format_exc())

    return return_parameter, return_remote_memory, history


def _train(
    config: Config,
    init_parameter: Optional[RLParameter],
    init_remote_memory: Optional[RLRemoteMemory],
    manager: Any,
    disable_trainer: bool,
    return_memory: bool,
    save_memory: str,
):
    # callbacks
    _info = {
        "config": config,
    }
    [c.on_init(_info) for c in config.callbacks]

    # --- share values
    train_end_signal = cast(ctypes.c_bool, mp.Value(ctypes.c_bool, False))
    remote_memory = manager.RemoteMemory(config.rl_config)
    remote_board = manager.Board()

    # init
    if init_remote_memory is None:
        if os.path.isfile(config.rl_config.remote_memory_path):
            remote_memory.load(config.rl_config.remote_memory_path)
    else:
        remote_memory.restore(init_remote_memory.backup())
    if init_parameter is None:
        _parameter = config.make_parameter()
        remote_board.write(_parameter.backup())
    else:
        remote_board.write(init_parameter.backup())

    # --- actor
    actors_ps_list = []
    for actor_id in range(config.actor_num):
        params = (
            config,
            remote_memory,
            remote_board,
            actor_id,
            train_end_signal,
        )
        ps = mp.Process(target=__run_actor, args=params)
        actors_ps_list.append(ps)

    # --- trainer
    if disable_trainer:
        trainer_ps = None
    else:
        params = (
            config,
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
    [c.on_start(_info) for c in config.callbacks]

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
        [c.on_polling(_info) for c in config.callbacks]

        if train_end_signal.value:
            break
    logger.info(f"wait loop end.(run time: {(time.time() - t0)/60:.2f}m)")

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

    return_parameter = config.make_parameter()
    return_remote_memory = config.make_remote_memory()
    try:
        # --- last parameter
        t0 = time.time()
        params = remote_board.read()
        if params is not None:
            return_parameter.restore(params)
        logger.info(f"recv parameter time: {time.time() - t0:.1f}s")

        # --- last memory
        if save_memory != "":
            remote_memory.save(save_memory, compress=True)
        if return_memory:
            t0 = time.time()
            return_remote_memory.restore(remote_memory.backup())
            logger.info(f"recv remote_memory time: {time.time() - t0:.1f}s")

        # callbacks
        [c.on_end(_info) for c in config.callbacks]
    except Exception:
        logger.warning(traceback.format_exc())

    return return_parameter, return_remote_memory
