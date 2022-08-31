import logging
import random
import time
import traceback
from typing import List, Optional, Tuple, Union, cast

import numpy as np
from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory
from srl.runner import sequence
from srl.runner.callback import TrainerCallback
from srl.runner.callbacks.file_logger import FileLogger, FileLogPlot
from srl.runner.callbacks.print_progress import TrainerPrintProgress
from srl.runner.sequence import Config
from srl.utils.common import is_package_installed

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
    eval_interval: int = 0,  # episode
    eval_num_episode: int = 1,
    eval_players: List[Union[None, str, RLConfig]] = [],
    eval_player: int = 0,
    # PrintProgress
    print_progress: bool = True,
    progress_max_time: int = 60 * 10,  # s
    progress_start_time: int = 5,
    progress_print_train_info: bool = True,
    # history
    enable_file_logger: bool = True,
    file_logger_tmp_dir: str = "tmp",
    file_logger_interval: int = 1,  # s
    enable_checkpoint: bool = True,
    checkpoint_interval: int = 60 * 20,  # s
    # other
    callbacks: List[TrainerCallback] = [],
) -> Tuple[RLParameter, RLRemoteMemory, object]:
    eval_players = eval_players[:]
    callbacks = callbacks[:]

    assert max_train_count > 0 or timeout > 0, "Please specify 'max_train_count' or 'timeout'."

    config = config.copy(env_share=False)
    # stop config
    config.max_train_count = max_train_count
    config.timeout = timeout
    # play config
    if config.seed is None:
        config.seed = seed
    # evaluate
    config.enable_evaluation = enable_evaluation
    config.eval_interval = eval_interval
    config.eval_num_episode = eval_num_episode
    config.eval_players = eval_players
    config.eval_player = eval_player
    # callbacks
    config.callbacks.extend(callbacks)
    # play info
    config.training = True
    config.distributed = False

    # PrintProgress
    if print_progress:
        config.callbacks.append(
            TrainerPrintProgress(
                max_time=progress_max_time,
                start_time=progress_start_time,
                print_train_info=progress_print_train_info,
            )
        )

    # FileLogger
    if enable_file_logger:
        file_logger = FileLogger(
            tmp_dir=file_logger_tmp_dir,
            enable_log=True,
            log_interval=file_logger_interval,
            enable_checkpoint=enable_checkpoint,
            checkpoint_interval=checkpoint_interval,
        )
        config.callbacks.append(file_logger)
    else:
        file_logger = None

    # play
    parameter, remote_memory = play(config, parameter, remote_memory)

    # history
    history = FileLogPlot()
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

        if is_package_installed("tensorflow"):
            import tensorflow as tf

            tf.random.set_seed(config.seed)

    # --- config
    config = config.copy(env_share=True)
    config.assert_params()

    # --- trainer
    if parameter is None:
        parameter = config.make_parameter()
    if remote_memory is None:
        remote_memory = config.make_remote_memory()
    trainer = config.make_trainer(parameter, remote_memory)
    callbacks = [c for c in config.callbacks if issubclass(c.__class__, TrainerCallback)]
    callbacks = cast(List[TrainerCallback], callbacks)

    # --- eval
    if config.enable_evaluation:
        eval_config = config.copy(env_share=False)
        eval_config.enable_evaluation = False
        eval_config.players = config.eval_players
        eval_config.rl_config.remote_memory_path = ""
        env = eval_config.make_env()
    else:
        env = None

    # callback
    _info = {
        "config": config,
        "parameter": parameter,
        "remote_memory": remote_memory,
        "trainer": trainer,
        "env": env,
        "train_count": 0,
    }
    [c.on_trainer_start(_info) for c in callbacks]

    # --- init
    t0 = time.time()
    end_reason = ""
    train_count = 0

    # --- loop
    while True:
        _time = time.time()

        # stop check
        if config.timeout > 0 and _time - t0 > config.timeout:
            end_reason = "timeout."
            break

        if config.max_train_count > 0 and train_count > config.max_train_count:
            end_reason = "max_train_count over."
            break

        # train
        train_t0 = _time
        train_info = trainer.train()
        train_time = time.time() - train_t0
        train_count = trainer.get_train_count()

        # eval
        eval_reward = None
        if config.enable_evaluation:
            if train_count % (config.eval_interval + 1) == 0:
                rewards = sequence.evaluate(
                    eval_config,
                    parameter=parameter,
                    max_episodes=config.eval_num_episode,
                )
                if env.player_num > 1:
                    rewards = [r[config.eval_player] for r in rewards]
                eval_reward = np.mean(rewards)

        # callbacks
        _info["train_info"] = train_info
        _info["train_time"] = train_time
        _info["train_count"] = train_count
        _info["eval_reward"] = eval_reward
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
