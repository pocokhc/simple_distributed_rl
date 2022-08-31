import ctypes
import logging
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass
from multiprocessing.managers import BaseManager
from typing import List, Optional, Tuple, Union

from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory
from srl.base.rl.registration import make_remote_memory
from srl.runner import sequence
from srl.runner.callback import Callback, MPCallback, TrainerCallback
from srl.runner.callbacks.file_logger import FileLogger, FileLogPlot
from srl.runner.callbacks.print_progress import MPPrintProgress, PrintProgress, TrainerPrintProgress
from srl.runner.trainer import play as trainer_play
from srl.utils.common import is_package_installed

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

→ メイン関数に "if __name__ == '__main__':" を明示していないと表示されます。
"""


# --------------------
# Config
# --------------------
@dataclass
class MpConfig:

    actor_num: int

    trainer_parameter_send_interval_by_train_count: int = 100
    actor_parameter_sync_interval_by_step: int = 100

    use_tensorflow: Optional[bool] = None
    allocate_trainer: str = "/CPU:0"
    allocate_actor: Union[List[str], str] = "/CPU:0"

    def __post_init__(self):
        # callbacks
        self.callbacks: List[MPCallback] = []

        if self.use_tensorflow is None:
            if is_package_installed("tensorflow"):
                self.use_tensorflow = True
            else:
                self.use_tensorflow = False

    def assert_params(self):
        assert self.actor_num > 0

    # -------------------------------------

    def to_dict(self) -> dict:
        # TODO: list
        conf = {}
        for k, v in self.__dict__.items():
            if v is None or type(v) in [int, float, bool, str]:
                conf[k] = v
        return conf

    def copy(self):
        config = MpConfig(0)
        for k, v in self.__dict__.items():
            if v is None or type(v) in [int, float, bool, str]:
                setattr(config, k, v)

        config.callbacks = self.callbacks  # sync
        return config


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
class _ActorInterrupt(sequence.Callback):
    def __init__(
        self,
        remote_board: Board,
        parameter: RLParameter,
        train_end_signal: ctypes.c_bool,
        mp_config: MpConfig,
    ) -> None:
        self.remote_board = remote_board
        self.parameter = parameter
        self.train_end_signal = train_end_signal
        self.actor_parameter_sync_interval_by_step = mp_config.actor_parameter_sync_interval_by_step

        self.step = 0
        self.prev_update_count = 0
        self.sync_count = 0

    def on_trainer_start(self, info):
        info["sync"] = 0  # 情報追加

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


def _run_actor(
    config: sequence.Config,
    mp_config: MpConfig,
    remote_memory: RLRemoteMemory,
    remote_board: Board,
    actor_id: int,
    train_end_signal: ctypes.c_bool,
    allocate: str,
):
    if mp_config.use_tensorflow:
        import tensorflow as tf

        with tf.device(allocate):
            __run_actor(config, mp_config, remote_memory, remote_board, actor_id, train_end_signal)
    else:
        __run_actor(config, mp_config, remote_memory, remote_board, actor_id, train_end_signal)


def __run_actor(
    config: sequence.Config,
    mp_config: MpConfig,
    remote_memory: RLRemoteMemory,
    remote_board: Board,
    actor_id: int,
    train_end_signal: ctypes.c_bool,
):
    try:
        logger.debug(f"actor{actor_id} start")

        config.disable_trainer = True
        config.rl_config.set_config_by_actor(mp_config.actor_num, actor_id)

        # eval するのはactor0のみ
        if actor_id != 0:
            config.enable_evaluation = False

        # parameter
        parameter = config.make_parameter(is_load=False)
        params = remote_board.read()
        if params is not None:
            parameter.restore(params)

        # callbacks
        config.callbacks.append(
            _ActorInterrupt(
                remote_board,
                parameter,
                train_end_signal,
                mp_config,
            )
        )

        # play
        sequence.play(config, parameter, remote_memory, actor_id)

    finally:
        train_end_signal.value = True
        logger.info(f"actor{actor_id} end")


# --------------------
# trainer
# --------------------
class _TrainerInterrupt(sequence.TrainerCallback):
    def __init__(
        self,
        remote_board: Board,
        parameter: RLParameter,
        train_end_signal: ctypes.c_bool,
        mp_config: MpConfig,
    ) -> None:
        self.remote_board = remote_board
        self.parameter = parameter
        self.train_end_signal = train_end_signal
        self.mp_config = mp_config

        self.sync_count = 0

    def on_trainer_start(self, info):
        info["sync"] = 0  # 情報追加

    def on_trainer_train(self, info):
        train_count = info["train_count"]

        if train_count == 0:
            time.sleep(1)
            return

        if train_count % self.mp_config.trainer_parameter_send_interval_by_train_count == 0:
            self.remote_board.write(self.parameter.backup())
            self.sync_count += 1
            info["sync"] = self.sync_count

    def intermediate_stop(self, info) -> bool:
        if self.train_end_signal.value:
            return True
        return False


def _run_trainer(
    config: sequence.Config,
    mp_config: MpConfig,
    remote_memory: RLRemoteMemory,
    remote_board: Board,
    train_end_signal: ctypes.c_bool,
):

    if mp_config.use_tensorflow:
        import tensorflow as tf

        with tf.device(mp_config.allocate_trainer):
            __run_trainer(config, mp_config, remote_memory, remote_board, train_end_signal)
    else:
        __run_trainer(config, mp_config, remote_memory, remote_board, train_end_signal)


def __run_trainer(
    config: sequence.Config,
    mp_config: MpConfig,
    remote_memory: RLRemoteMemory,
    remote_board: Board,
    train_end_signal: ctypes.c_bool,
):
    logger.debug("trainer start")

    # parameter
    parameter = config.make_parameter(is_load=False)
    params = remote_board.read()
    if params is not None:
        parameter.restore(params)

    try:
        config.enable_evaluation = False

        # callbacks
        config.callbacks.append(_TrainerInterrupt(remote_board, parameter, train_end_signal, mp_config))

        # play
        trainer_play(config, parameter, remote_memory)

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


def train(
    config: sequence.Config,
    mp_config: MpConfig,
    # stop config
    max_episodes: int = -1,
    timeout: int = -1,
    max_steps: int = -1,
    max_train_count: int = -1,
    # play config
    shuffle_player: bool = True,
    disable_trainer: bool = False,
    # evaluate
    enable_evaluation: bool = True,
    eval_interval: int = 0,  # episode
    eval_num_episode: int = 1,
    eval_players: List[Union[None, str, RLConfig]] = [],
    eval_player: int = 0,
    # PrintProgress
    print_progress: bool = True,
    progress_max_time: int = 60 * 10,  # s
    progress_start_time: int = 5,
    progress_print_env_info: bool = False,
    progress_print_worker_info: bool = True,
    progress_print_train_info: bool = True,
    progress_print_worker: int = 0,
    progress_max_actor: int = 5,
    # history
    enable_file_logger: bool = True,
    file_logger_tmp_dir: str = "tmp",
    file_logger_interval: int = 1,  # s
    enable_checkpoint: bool = True,
    checkpoint_interval: int = 60 * 20,  # s
    # other
    callbacks: List[Union[Callback, TrainerCallback]] = [],
    mp_callbacks: List[MPCallback] = [],
    init_parameter: Optional[RLParameter] = None,
    init_remote_memory: Optional[RLRemoteMemory] = None,
    return_memory: bool = False,
    save_memory: str = "",
) -> Tuple[RLParameter, RLRemoteMemory, FileLogPlot]:
    eval_players = eval_players[:]
    callbacks = callbacks[:]
    mp_callbacks = mp_callbacks[:]

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
    mp_config = mp_config.copy()
    # stop config
    config.max_episodes = max_episodes
    config.timeout = timeout
    config.max_steps = max_steps
    config.max_train_count = max_train_count
    # play config
    config.shuffle_player = shuffle_player
    config.disable_trainer = disable_trainer
    # evaluate
    config.enable_evaluation = enable_evaluation
    config.eval_interval = eval_interval
    config.eval_num_episode = eval_num_episode
    config.eval_players = eval_players
    config.eval_player = eval_player
    # callbacks
    config.callbacks = callbacks
    mp_config.callbacks = mp_callbacks
    # play info
    config.training = True
    config.distributed = True

    # PrintProgress
    if print_progress:
        config.callbacks.append(
            PrintProgress(
                max_time=progress_max_time,
                start_time=progress_start_time,
                print_env_info=progress_print_env_info,
                print_worker_info=progress_print_worker_info,
                print_train_info=False,
                print_worker=progress_print_worker,
                max_actor=progress_max_actor,
            )
        )
        config.callbacks.append(
            TrainerPrintProgress(
                max_time=progress_max_time,
                start_time=progress_start_time,
                print_train_info=progress_print_train_info,
            )
        )
        mp_config.callbacks.append(MPPrintProgress())

    if enable_file_logger:
        file_logger = FileLogger(
            tmp_dir=file_logger_tmp_dir,
            enable_log=True,
            log_interval=file_logger_interval,
            enable_checkpoint=enable_checkpoint,
            checkpoint_interval=checkpoint_interval,
        )
        config.callbacks.append(file_logger)
        mp_config.callbacks.append(file_logger)
    else:
        file_logger = None

    # ---
    config.assert_params()
    mp_config.assert_params()

    remote_memory_class = make_remote_memory(config.rl_config, return_class=True)

    # mp を notebook で実行する場合はrlの定義をpyファイルにする必要あり TODO: それ以外でも動かないような
    # if is_env_notebook() and "__main__" in str(remote_memory_class):
    #    raise RuntimeError("The definition of rl must be in the py file")

    MPManager.register("RemoteMemory", remote_memory_class)
    MPManager.register("Board", Board)

    with MPManager() as manager:
        return_parameter, return_remote_memory = _train(
            config,
            mp_config,
            init_parameter,
            init_remote_memory,
            manager,
            disable_trainer,
            return_memory,
            save_memory,
        )

    # history
    history = FileLogPlot()
    try:
        if file_logger is not None:
            history.load(file_logger.base_dir)
    except Exception:
        logger.warning(traceback.format_exc())
    return return_parameter, return_remote_memory, history


def _train(
    config: sequence.Config,
    mp_config: MpConfig,
    init_parameter: Optional[RLParameter],
    init_remote_memory: Optional[RLRemoteMemory],
    manager: MPManager,
    disable_trainer: bool,
    return_memory: bool,
    save_memory: str,
):
    enable_trainer = not disable_trainer

    # callbacks
    _info = {
        "config": config,
        "mp_config": mp_config,
    }
    [c.on_init(_info) for c in mp_config.callbacks]

    # --- share values
    train_end_signal = mp.Value(ctypes.c_bool, False)
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
    for actor_id in range(mp_config.actor_num):
        if isinstance(mp_config.allocate_actor, str):
            allocate = mp_config.allocate_actor
        else:
            allocate = mp_config.allocate_actor[actor_id]
        params = (
            config,
            mp_config,
            remote_memory,
            remote_board,
            actor_id,
            train_end_signal,
            allocate,
        )
        ps = mp.Process(target=_run_actor, args=params)
        actors_ps_list.append(ps)

    # --- trainer
    if enable_trainer:
        params = (
            config,
            mp_config,
            remote_memory,
            remote_board,
            train_end_signal,
        )
        trainer_ps = mp.Process(target=_run_trainer, args=params)

    # --- start
    [p.start() for p in actors_ps_list]
    if enable_trainer:
        trainer_ps.start()

    # callbacks
    [c.on_start(_info) for c in mp_config.callbacks]

    # 終了を待つ
    while True:
        time.sleep(1)  # polling time

        # プロセスが落ちたら終了
        if enable_trainer:
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
        [c.on_polling(_info) for c in mp_config.callbacks]

        if train_end_signal.value:
            break

    # --- プロセスの終了を待つ
    for w in actors_ps_list:
        for _ in range(5):
            if w.is_alive():
                time.sleep(1)
            else:
                break
        else:
            w.terminate()
    if enable_trainer:
        for _ in range(60 * 10):
            if trainer_ps.is_alive():
                time.sleep(1)
            else:
                break
        else:
            trainer_ps.terminate()

    # 子プロセスが正常終了していなければ例外を出す
    # exitcode: 0 正常, 1 例外, 負 シグナル
    if enable_trainer and trainer_ps.exitcode != 0:
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
        [c.on_end(_info) for c in mp_config.callbacks]
    except Exception:
        logger.warning(traceback.format_exc())

    return return_parameter, return_remote_memory
