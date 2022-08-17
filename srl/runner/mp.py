import ctypes
import logging
import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass
from multiprocessing.managers import BaseManager
from typing import Dict, List, Optional, Tuple, Union

import tensorflow as tf
from srl.base.rl.base import RLParameter, RLRemoteMemory
from srl.base.rl.registration import make_remote_memory
from srl.runner import sequence
from srl.runner.callback_mp import MPCallback
from srl.runner.callbacks_mp.mp_file_logger import MPFileLogger
from srl.runner.callbacks_mp.mp_print_progress import MPPrintProgress
from srl.runner.file_log_plot import FileLogPlot

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
class Config:

    actor_num: int

    trainer_parameter_send_interval_by_train_count: int = 100
    actor_parameter_sync_interval_by_step: int = 100

    allocate_main: str = "/CPU:0"
    allocate_trainer: str = "/CPU:0"
    allocate_actor: Union[List[str], str] = "/CPU:0"

    def __post_init__(self):
        # train config
        self.max_train_count: int = -1
        self.timeout: int = -1
        # callbacks
        self.callbacks: List[MPCallback] = []

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
        config = Config(0)
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
        mp_config: Config,
    ) -> None:
        self.remote_board = remote_board
        self.parameter = parameter
        self.train_end_signal = train_end_signal
        self.actor_parameter_sync_interval_by_step = mp_config.actor_parameter_sync_interval_by_step

        self.timeout = mp_config.timeout

        self.t0 = time.time()
        self.step = 0
        self.prev_update_count = 0

    def on_step_end(self, **kwargs):
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

    def intermediate_stop(self, **kwargs) -> bool:
        if self.train_end_signal.value:
            return True

        # timeout は時差を少なくするために main ではなく actor でチェック
        elapsed_time = time.time() - self.t0
        if self.timeout > 0 and elapsed_time > self.timeout:
            return True
        return False


def _run_actor(
    config: sequence.Config,
    mp_config: Config,
    remote_memory: RLRemoteMemory,
    remote_board: Board,
    actor_id: int,
    train_end_signal: ctypes.c_bool,
    allocate: str,
):
    with tf.device(allocate):
        try:
            logger.debug(f"actor{actor_id} start")

            config.callbacks.extend(mp_config.callbacks)
            config.disable_trainer = True
            config.rl_config.set_config_by_actor(mp_config.actor_num, actor_id)

            # validationするのはactor0のみ
            if actor_id != 0:
                config.enable_validation = False

            # parameter
            parameter = config.make_parameter()
            params = remote_board.read()
            if params is not None:
                parameter.restore(params)
            config.callbacks.append(
                _ActorInterrupt(
                    remote_board,
                    parameter,
                    train_end_signal,
                    mp_config,
                )
            )

            # train
            sequence.play(config, parameter, remote_memory, actor_id)

        finally:
            train_end_signal.value = True
            logger.info(f"actor{actor_id} end")


# --------------------
# trainer
# --------------------
def _run_trainer(
    config: sequence.Config,
    mp_config: Config,
    remote_memory: RLRemoteMemory,
    remote_board: Board,
    train_end_signal: ctypes.c_bool,
):
    with tf.device(mp_config.allocate_trainer):
        logger.debug("trainer start")

        parameter = config.make_parameter()
        params = remote_board.read()
        if params is not None:
            parameter.restore(params)
        trainer = config.make_trainer(parameter, remote_memory)

        # loop var
        callbacks = [c for c in mp_config.callbacks if issubclass(c.__class__, MPCallback)]
        sync_count = 0

        # callbacks
        _info = {
            "config": config,
            "mp_config": mp_config,
            "trainer": trainer,
            "parameter": parameter,
            "remote_memory": remote_memory,
            "train_count": 0,
        }
        [c.on_trainer_start(**_info) for c in callbacks]

        try:
            while True:
                if train_end_signal.value:
                    break

                if mp_config.max_train_count > 0 and trainer.get_train_count() > mp_config.max_train_count:
                    break

                train_t0 = time.time()
                train_info = trainer.train()
                train_time = time.time() - train_t0

                if trainer.get_train_count() == 0:
                    time.sleep(1)
                else:
                    # send parameter
                    if trainer.get_train_count() % mp_config.trainer_parameter_send_interval_by_train_count == 0:
                        remote_board.write(parameter.backup())
                        sync_count += 1

                # callbacks
                _info["train_info"] = train_info
                _info["train_count"] = trainer.get_train_count()
                _info["train_time"] = train_time
                _info["sync_count"] = sync_count
                [c.on_trainer_train(**_info) for c in callbacks]

        finally:
            train_end_signal.value = True
            t0 = time.time()
            remote_board.write(parameter.backup())
            logger.info(f"trainer end.(send parameter time: {time.time() - t0:.1f}s)")

            # callbacks
            [c.on_trainer_end(**_info) for c in callbacks]


# ----------------------------
# 学習
# ----------------------------
class MPManager(BaseManager):
    pass


def train(
    config: sequence.Config,
    mp_config: Config,
    # train config
    max_train_count: int = -1,
    timeout: int = -1,
    shuffle_player: bool = True,
    enable_validation: bool = True,
    disable_trainer: bool = False,
    # print
    print_progress: bool = True,
    max_progress_time: int = 60 * 10,  # s
    print_progress_kwargs: Optional[Dict] = None,
    # log
    enable_file_logger: bool = True,
    file_logger_kwargs: Optional[Dict] = None,
    remove_file_logger: bool = True,
    # other
    callbacks: List[MPCallback] = None,
    init_parameter: Optional[RLParameter] = None,
    init_remote_memory: Optional[RLRemoteMemory] = None,
    return_memory: bool = False,
    save_memory: str = "",
) -> Tuple[RLParameter, RLRemoteMemory, FileLogPlot]:
    if callbacks is None:
        callbacks = []
    if disable_trainer:
        enable_validation = False  # 学習しないので
        assert timeout != -1, "Please specify 'timeout'."

    assert max_train_count != -1 or timeout != -1, "Please specify 'max_train_count' or 'timeout'."

    config = config.copy(env_share=False)
    mp_config = mp_config.copy()

    config.shuffle_player = shuffle_player
    config.enable_validation = enable_validation
    config.training = True
    config.distributed = True

    mp_config.max_train_count = max_train_count
    mp_config.timeout = timeout

    if print_progress:
        if print_progress_kwargs is None:
            mp_config.callbacks.append(MPPrintProgress(max_progress_time=max_progress_time))
        else:
            mp_config.callbacks.append(MPPrintProgress(max_progress_time=max_progress_time, **print_progress_kwargs))

    if file_logger_kwargs is None:
        file_logger = MPFileLogger()
    else:
        file_logger = MPFileLogger(**file_logger_kwargs)
    if enable_file_logger:
        mp_config.callbacks.append(file_logger)

    config.assert_params()
    mp_config.assert_params()

    with tf.device(mp_config.allocate_main):

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

    try:
        history = FileLogPlot()
        if enable_file_logger:
            history.load(file_logger.base_dir, remove_file_logger)
        return return_parameter, return_remote_memory, history
    except Exception:
        logger.warning(traceback.format_exc())
    return return_parameter, return_remote_memory, None


def _train(
    config: sequence.Config,
    mp_config: Config,
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
    [c.on_init(**_info) for c in mp_config.callbacks]

    # --- share values
    train_end_signal = mp.Value(ctypes.c_bool, False)
    remote_memory = manager.RemoteMemory(config.rl_config)
    remote_board = manager.Board()

    # init
    if init_remote_memory is not None:
        remote_memory.restore(init_remote_memory.backup())
    if init_parameter is not None:
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
    [c.on_start(**_info) for c in mp_config.callbacks]

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
        [c.on_polling(**_info) for c in mp_config.callbacks]

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

    return_parameter = None
    return_remote_memory = None
    try:
        # --- last parameter
        t0 = time.time()
        return_parameter = config.make_parameter()
        params = remote_board.read()
        if params is not None:
            return_parameter.restore(params)
        logger.info(f"recv parameter time: {time.time() - t0:.1f}s")

        # --- last memory
        if save_memory != "":
            remote_memory.save(save_memory, compress=True)
        if return_memory:
            t0 = time.time()
            return_remote_memory = config.make_remote_memory()
            return_remote_memory.restore(remote_memory.backup())
            logger.info(f"recv remote_memory time: {time.time() - t0:.1f}s")

        # callbacks
        [c.on_end(**_info) for c in mp_config.callbacks]
    except Exception:
        logger.warning(traceback.format_exc())

    return return_parameter, return_remote_memory
