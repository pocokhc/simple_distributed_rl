import logging
import random
import time
import traceback
from typing import List, Optional, Tuple, Union

import numpy as np
from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory
from srl.runner.callback import Callback
from srl.runner.sequence import Config
from srl.utils.common import is_package_imported

logger = logging.getLogger(__name__)


# ---------------------------------
# train
# ---------------------------------
def train(
    config: Config,
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
    # stop config
    max_train_count: int = -1,
    timeout: int = -1,
    # play config
    seed: Optional[int] = None,
    # evaluate
    enable_evaluation: bool = False,
    eval_env_sharing: bool = True,
    eval_interval: int = 0,  # episode
    eval_num_episode: int = 1,
    eval_players: List[Union[None, str, RLConfig]] = [],
    # PrintProgress
    print_progress: bool = True,
    progress_max_time: int = 60 * 10,  # s
    progress_start_time: int = 5,
    progress_print_train_info: bool = True,
    # history
    enable_file_logger: bool = True,
    file_logger_tmp_dir: str = "tmp",
    file_logger_enable_train_log: bool = True,
    file_logger_train_log_interval: int = 1,  # s
    file_logger_enable_checkpoint: bool = True,
    file_logger_checkpoint_interval: int = 60 * 20,  # s
    # other
    callbacks: List[Callback] = [],
):
    assert max_train_count > 0 or timeout > 0, "Please specify 'max_train_count' or 'timeout'."

    config = config.copy(env_share=True)
    # stop config
    config.max_train_count = max_train_count
    config.timeout = timeout
    # play config
    if config.seed is None:
        config.seed = seed
    # callbacks
    config.callbacks = callbacks[:]
    # play info
    config.training = True
    config.distributed = False

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
        from srl.runner.callbacks.print_progress import PrintProgress

        config.callbacks.append(
            PrintProgress(
                max_time=progress_max_time,
                start_time=progress_start_time,
                print_train_info=progress_print_train_info,
            )
        )

    # --- FileLog
    if enable_file_logger:
        from srl.runner.callbacks.file_log_writer import FileLogWriter

        file_logger = FileLogWriter(
            tmp_dir=file_logger_tmp_dir,
            enable_train_log=file_logger_enable_train_log,
            enable_episode_log=False,
            train_log_interval=file_logger_train_log_interval,
            enable_checkpoint=file_logger_enable_checkpoint,
            checkpoint_interval=file_logger_checkpoint_interval,
        )
        config.callbacks.append(file_logger)
    else:
        file_logger = None

    # --- play
    parameter, remote_memory = play(config, parameter, remote_memory)

    # --- history
    from srl.runner.callbacks.file_log_reader import FileLogReader

    history = FileLogReader()
    try:
        if file_logger is not None:
            history.load(file_logger.base_dir)
    except Exception:
        logger.warning(traceback.format_exc())

    return parameter, remote_memory, history


# ---------------------------------
# play main
# ---------------------------------
def play(
    config: Config,
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
) -> Tuple[RLParameter, RLRemoteMemory]:

    # --- random seed
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)

        if is_package_imported("tensorflow"):
            import tensorflow as tf

            tf.random.set_seed(config.seed)

    # --- config
    config = config.copy(env_share=True)
    config.assert_params()

    # --- parameter/remote_memory/trainer
    if parameter is None:
        parameter = config.make_parameter()
    if remote_memory is None:
        remote_memory = config.make_remote_memory()
    trainer = config.make_trainer(parameter, remote_memory)

    # callbacks
    callbacks = [c for c in config.callbacks if issubclass(c.__class__, Callback)]

    # callbacks
    _info = {
        "config": config,
        "parameter": parameter,
        "remote_memory": remote_memory,
        "trainer": trainer,
        "train_count": 0,
    }
    [c.on_trainer_start(_info) for c in callbacks]

    # --- init
    t0 = time.time()
    end_reason = ""
    train_count = 0

    # --- loop
    while True:
        train_t0 = time.time()

        # stop check
        if config.timeout > 0 and train_t0 - t0 > config.timeout:
            end_reason = "timeout."
            break

        if config.max_train_count > 0 and train_count > config.max_train_count:
            end_reason = "max_train_count over."
            break

        # train
        train_info = trainer.train()
        train_time = time.time() - train_t0
        train_count = trainer.get_train_count()

        # callbacks
        _info["train_info"] = train_info
        _info["train_time"] = train_time
        _info["train_count"] = train_count
        [c.on_trainer_train(_info) for c in callbacks]

        # callback end
        if True in [c.intermediate_stop(_info) for c in callbacks]:
            end_reason = "callback.intermediate_stop"
            break

    # callbacks
    _info["train_count"] = train_count
    _info["end_reason"] = end_reason
    [c.on_trainer_end(_info) for c in callbacks]

    return parameter, remote_memory
