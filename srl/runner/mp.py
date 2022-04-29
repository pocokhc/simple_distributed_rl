import ctypes
import logging
import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing.managers import BaseManager
from typing import List, Optional, Tuple, Union

import tensorflow as tf
from srl.base.rl.base import RLParameter, RLRemoteMemory
from srl.base.rl.registration import make_remote_memory
from srl.runner import sequence
from srl.runner.callbacks_mp import MPCallback

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

    worker_num: int
    max_train_count: int = -1
    timeout: int = -1

    trainer_parameter_send_interval_by_train_count: int = 100
    worker_parameter_sync_interval_by_step: int = 10

    parameter_send_timeout: int = 60 * 10  # s

    allocate_main: str = "/CPU:0"
    allocate_trainer: str = "/GPU:0"
    allocate_worker: Union[List[str], str] = "/CPU:0"

    callbacks: List[MPCallback] = field(default_factory=list)

    def set_train_config(
        self,
        max_train_count: int = -1,
        timeout: int = -1,
        callbacks: List[MPCallback] = None,
    ):
        if callbacks is None:
            callbacks = []
        self.max_train_count = max_train_count
        self.timeout = timeout
        self.callbacks = callbacks

    # -------------------------------------

    def to_dict(self) -> dict:
        conf = {}
        for k, v in self.__dict__.items():
            if type(v) in [int, float, bool, str]:
                conf[k] = v
        return conf

    def copy(self):
        config = Config(0)
        for k, v in self.__dict__.items():
            if type(v) in [int, float, bool, str]:
                setattr(config, k, v)
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
# worker
# --------------------
class _SyncParameter(sequence.Callback):
    def __init__(
        self,
        remote_board: Board,
        parameter: RLParameter,
        mp_config: Config,
    ) -> None:
        self.remote_board = remote_board
        self.parameter = parameter
        self.worker_parameter_sync_interval_by_step = mp_config.worker_parameter_sync_interval_by_step

        self.step = 0
        self.prev_update_count = 0

    def on_step_end(self, **kwargs):
        self.step += 1
        if self.step % self.worker_parameter_sync_interval_by_step != 0:
            return
        update_count = self.remote_board.get_update_count()
        if update_count == self.prev_update_count:
            return
        self.prev_update_count = update_count
        params = self.remote_board.read()
        if params is None:
            return
        self.parameter.restore(params)


class _InterruptEnd(sequence.Callback):
    def __init__(self, train_end_signal: ctypes.c_bool) -> None:
        self.train_end_signal = train_end_signal

    def intermediate_stop(self, **kwargs) -> bool:
        if self.train_end_signal.value:
            return True
        return False


def _run_worker(
    config: sequence.Config,
    mp_config: Config,
    remote_memory: RLRemoteMemory,
    remote_board: Board,
    worker_id: int,
    train_end_signal: ctypes.c_bool,
    allocate: str,
):
    with tf.device(allocate):
        logger.debug(f"worker{worker_id} start")

        try:
            config.callbacks.extend(mp_config.callbacks)
            config.callbacks.append(_InterruptEnd(train_end_signal))
            config.trainer_disable = True

            parameter = config.make_parameter()
            params = remote_board.read()
            if params is not None:
                parameter.restore(params)
            config.callbacks.append(_SyncParameter(remote_board, parameter, mp_config))

            sequence.play(config, parameter, remote_memory, None, worker_id)

        except Exception:
            logger.warn(traceback.format_exc())
        finally:
            train_end_signal.value = True
            logger.debug(f"worker{worker_id} end")


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
        callbacks = [c for c in mp_config.callbacks if issubclass(c.__class__, MPCallback)]
        sync_count = 0

        # callbacks
        _info = {
            "config": config,
            "mp_config": mp_config,
            "trainer": trainer,
            "parameter": parameter,
            "remote_memory": remote_memory,
        }
        [c.on_trainer_start(**_info) for c in callbacks]

        try:
            while True:
                if train_end_signal.value:
                    break

                if mp_config.max_train_count > 0 and trainer.get_train_count() >= mp_config.max_train_count:
                    break

                t0 = time.time()
                train_info = trainer.train()
                train_time = time.time() - t0

                if trainer.get_train_count() == 0:
                    time.sleep(1)
                    continue

                # send parameter
                if trainer.get_train_count() % mp_config.trainer_parameter_send_interval_by_train_count == 0:
                    remote_board.write(parameter.backup())
                    sync_count += 1

                # callbacks
                _info["time_"] = time.time()
                _info["train_info"] = train_info
                _info["train_count"] = trainer.get_train_count()
                _info["train_time"] = train_time
                _info["sync_count"] = sync_count
                [c.on_trainer_train_end(**_info) for c in callbacks]

        except Exception:
            logger.warn(traceback.format_exc())
        finally:
            train_end_signal.value = True

            remote_board.write(parameter.backup())
            logger.debug("trainer end")

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
    init_parameter: Optional[RLParameter] = None,
    init_remote_memory: Optional[RLRemoteMemory] = None,
    return_memory: bool = False,
) -> Tuple[RLParameter, RLRemoteMemory]:
    # TODO config copy

    with tf.device(mp_config.allocate_main):

        # config の初期化
        config.init_rl_config()
        MPManager.register("RemoteMemory", make_remote_memory(config.rl_config, None, get_class=True))
        MPManager.register("Board", Board)

        with MPManager() as manager:
            return_parameter, return_remote_memory = _train(
                config,
                mp_config,
                init_parameter,
                init_remote_memory,
                manager,
                return_memory,
            )

        return return_parameter, return_remote_memory


def _train(
    config: sequence.Config,
    mp_config: Config,
    init_parameter: Optional[RLParameter],
    init_remote_memory: Optional[RLRemoteMemory],
    manager: MPManager,
    return_memory: bool,
):
    # config
    config.training = True
    config.rl_config.assert_params()

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

    # --- worker
    workers_ps_list = []
    for worker_id in range(mp_config.worker_num):
        if isinstance(mp_config.allocate_worker, str):
            allocate = mp_config.allocate_worker
        else:
            allocate = mp_config.allocate_worker[worker_id]
        params = (
            config,
            mp_config,
            remote_memory,
            remote_board,
            worker_id,
            train_end_signal,
            allocate,
        )
        ps = mp.Process(target=_run_worker, args=params)
        workers_ps_list.append(ps)

    # --- trainer
    params = (
        config,
        mp_config,
        remote_memory,
        remote_board,
        train_end_signal,
    )
    trainer_ps = mp.Process(target=_run_trainer, args=params)

    # --- start
    [p.start() for p in workers_ps_list]
    trainer_ps.start()

    # callbacks
    [c.on_start(**_info) for c in mp_config.callbacks]

    # 終了を待つ
    try:
        t0 = time.time()
        while True:
            time.sleep(1)  # polling time

            # timeout
            _time = time.time()
            elapsed_time = _time - t0
            if mp_config.timeout > 0 and elapsed_time > mp_config.timeout:
                train_end_signal.value = True
                logger.info("train end(reason: timeout)")

            # プロセスが落ちたら終了
            if not trainer_ps.is_alive():
                train_end_signal.value = True
                logger.info("train end(reason: trainer_ps dead or max_train_count)")
            if False in [w.is_alive() for w in workers_ps_list]:
                train_end_signal.value = True
                logger.info("train end(reason: worker_ps dead)")

            # callbacks
            _info["time_"] = _time
            _info["elapsed_time"] = elapsed_time
            [c.on_polling(**_info) for c in mp_config.callbacks]

            if train_end_signal.value:
                break
    except Exception:
        logger.debug(traceback.format_exc())
        logger.info("train end(reason: error)")
    finally:
        train_end_signal.value = True

    # --- end, 終了前に少し待つ
    time.sleep(1)
    for w in workers_ps_list:
        for _ in range(10):
            if w.is_alive():
                time.sleep(1)
            else:
                break
        else:
            w.terminate()
    for _ in range(10):
        if trainer_ps.is_alive():
            time.sleep(1)
        else:
            break
    else:
        trainer_ps.terminate()

    # --- last parameter
    return_parameter = config.make_parameter()
    params = remote_board.read()
    if params is not None:
        return_parameter.restore(params)

    # --- last memory
    if return_memory:
        return_remote_memory = config.make_remote_memory()
        return_remote_memory.restore(remote_memory.backup())
    else:
        return_remote_memory = None

    # callbacks
    [c.on_end(**_info) for c in mp_config.callbacks]

    return return_parameter, return_remote_memory


if __name__ == "__main__":
    pass
