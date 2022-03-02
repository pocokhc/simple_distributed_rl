import ctypes
import logging
import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing.managers import BaseManager
from typing import Any, List, Union

import tensorflow as tf
from srl.base.rl.memory import Memory
from srl.base.rl.rl import RLParameter
from srl.rl.memory import registory
from srl.runner import sequence
from srl.runner.callbacks_mp import MPCallback

logger = logging.getLogger(__name__)


# --------------------
# Config
# --------------------
@dataclass
class Config:

    worker_num: int
    max_train_count: int = -1
    timeout: int = -1

    sync_parameter_interval: int = 100

    memory_recv_max: int = 1  # memoryが１度に受信する最大値
    memory_send_summarize_num: int = 100  # workerがまとめて送る数
    worker_wait_memory_num: int = 1000

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

    def on_step_end(self, info):
        self.step += 1
        if self.step % self.worker_parameter_sync_interval_by_step != 0:
            return
        update_count = self.remote_board.get_update_count()
        if update_count == self.prev_update_count:
            return
        params = self.remote_board.read()
        if params is None:
            return
        self.parameter.restore(params)
        self.prev_update_count = update_count


class _InterruptEnd(sequence.Callback):
    def __init__(self, train_end_signal: ctypes.c_bool) -> None:
        self.train_end_signal = train_end_signal

    def intermediate_stop(self, _params) -> bool:
        if self.train_end_signal.value:
            return True
        return False


def _run_worker(
    config: sequence.Config,
    mp_config: Config,
    remote_memory: Memory,
    remote_board: Board,
    worker_id: int,
    train_end_signal: ctypes.c_bool,
    allocate: str,
):
    with tf.device(allocate):
        logger.debug(f"worker{worker_id} start")
        config.worker_id = worker_id

        try:
            config.callbacks.append(_InterruptEnd(train_end_signal))
            config.trainer_disable = True

            parameter = config.create_parameter()
            config.callbacks.append(_SyncParameter(remote_board, parameter, mp_config))

            sequence.play(config, parameter, remote_memory)

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
    remote_memory: Memory,
    remote_board: Board,
    last_param_q: mp.Queue,
    train_end_signal: ctypes.c_bool,
):
    with tf.device(mp_config.allocate_trainer):
        logger.debug("trainer start")

        parameter = config.create_parameter()
        trainer = config.create_trainer(parameter)
        callbacks = [c for c in mp_config.callbacks if issubclass(c.__class__, MPCallback)]
        sync_cout = 0

        # callbacks
        _info = {
            "config": config,
            "mp_config": mp_config,
            "trainer": trainer,
            "parameter": parameter,
        }
        [c.on_trainer_start(_info) for c in callbacks]

        try:
            while True:
                if train_end_signal.value:
                    break

                if mp_config.max_train_count > 0 and trainer.train_count >= mp_config.max_train_count:
                    break

                train_info = trainer.train(remote_memory)

                if trainer.train_count == 0:
                    time.sleep(1)
                    continue

                # send parameter
                is_sync = trainer.train_count % mp_config.trainer_parameter_send_interval_by_train_count == 0
                if is_sync:
                    remote_board.write(parameter.backup())
                    sync_cout += 1

                # callbacks
                _info["is_sync"] = is_sync
                _info["train_info"] = train_info
                _info["sync_cout"] = sync_cout
                [c.on_trainer_train_end(_info) for c in callbacks]

        except Exception:
            logger.warn(traceback.format_exc())
        finally:
            train_end_signal.value = True

            # --- 最後に重さを送信
            logger.info("send weight...")
            last_param_q.put(parameter.backup(), timeout=mp_config.parameter_send_timeout)
            logger.info("send success...")
            logger.debug("trainer end")

            # callbacks
            [c.on_trainer_end(_info) for c in callbacks]


# ----------------------------
# 学習
# ----------------------------
class MPManager(BaseManager):
    pass


def train(config: sequence.Config, mp_config: Config):
    with tf.device(mp_config.allocate_main):

        MPManager.register("Memory", registory.get_class(config.memory_config))
        MPManager.register("Board", Board)

        with MPManager() as manager:
            return _train(config, mp_config, manager)


def _train(config: sequence.Config, mp_config: Config, manager):

    # config
    config.training = True
    config.init_rl_config()
    config.callbacks.extend(mp_config.callbacks)

    # callbacks
    _info = {
        "config": config,
        "mp_config": mp_config,
    }
    [c.on_init(_info) for c in mp_config.callbacks]

    # --- share values
    train_end_signal = mp.Value(ctypes.c_bool, False)
    remote_memory = manager.Memory(config.memory_config)
    remote_board = manager.Board()
    last_param_q = mp.Queue()  # trainer -> main(last)

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
        last_param_q,
        train_end_signal,
    )
    trainer_ps = mp.Process(target=_run_trainer, args=params)

    # --- start
    [p.start() for p in workers_ps_list]
    trainer_ps.start()

    # callbacks
    _info = {
        "config": config,
        "mp_config": mp_config,
    }
    [c.on_start(_info) for c in mp_config.callbacks]

    # 終了を待つ
    try:
        t0 = time.time()
        while True:
            time.sleep(1)  # polling time

            # timeout
            elapsed_time = time.time() - t0
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
            _info = {
                "config": config,
                "mp_config": mp_config,
                "elapsed_time": elapsed_time,
            }
            [c.on_polling(_info) for c in mp_config.callbacks]

            if train_end_signal.value:
                break
    except Exception:
        logger.debug(traceback.format_exc())
        logger.info("train end(reason: error)")
    finally:
        train_end_signal.value = True

    # --- recv last param
    t0 = time.time()
    logger.info("wait param...")
    param = last_param_q.get(timeout=mp_config.parameter_send_timeout)
    parameter = config.create_parameter()
    parameter.restore(param)
    param = None
    last_param_q.close()
    logger.info(f"success({time.time()-t0:.2f}s)")

    # --- end
    time.sleep(1)  # 終了前に少し待つ
    for w in workers_ps_list:
        if w.is_alive():
            time.sleep(5)  # 強制終了前に少し待つ
            w.terminate()
    if trainer_ps.is_alive():
        time.sleep(5)  # 強制終了前に少し待つ
        trainer_ps.terminate()

    # callbacks
    _info = {
        "config": config,
        "mp_config": mp_config,
    }
    [c.on_end(_info) for c in mp_config.callbacks]

    return parameter


if __name__ == "__main__":
    pass
