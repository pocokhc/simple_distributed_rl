import logging
import traceback
from typing import List, Optional, Tuple, Union

from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory
from srl.runner.callback import Callback
from srl.runner.callbacks.file_logger import FileLogger, FileLogPlot
from srl.runner.callbacks.print_progress import PrintProgress
from srl.runner.callbacks.rendering import Rendering
from srl.runner.config import Config
from srl.runner.sequence_play import play

logger = logging.getLogger(__name__)


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
    seed: Optional[int] = None,
    # evaluate
    enable_evaluation: bool = True,
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
    # history
    enable_file_logger: bool = True,
    file_logger_tmp_dir: str = "tmp",
    file_logger_interval: int = 1,  # s
    enable_checkpoint: bool = True,
    checkpoint_interval: int = 60 * 20,  # s
    # other
    callbacks: List[Callback] = [],
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
) -> Tuple[RLParameter, RLRemoteMemory, FileLogPlot]:
    eval_players = eval_players[:]
    callbacks = callbacks[:]

    if disable_trainer:
        enable_evaluation = False  # 学習しないので
        assert (
            max_steps != -1 or max_episodes != -1 or timeout != -1
        ), "Please specify 'max_episodes', 'timeout' or 'max_steps'."
    else:
        assert (
            max_steps != -1 or max_episodes != -1 or timeout != -1 or max_train_count != -1
        ), "Please specify 'max_episodes', 'timeout' , 'max_steps' or 'max_train_count'."

    config = config.copy(env_share=True)
    # stop config
    config.max_episodes = max_episodes
    config.timeout = timeout
    config.max_steps = max_steps
    config.max_train_count = max_train_count
    # play config
    config.shuffle_player = shuffle_player
    config.disable_trainer = disable_trainer
    if config.seed is None:
        config.seed = seed
    # evaluate
    config.enable_evaluation = enable_evaluation
    config.eval_interval = eval_interval
    config.eval_num_episode = eval_num_episode
    config.eval_players = eval_players
    # callbacks
    config.callbacks = callbacks
    # play info
    config.training = True
    config.distributed = False

    # PrintProgress
    if print_progress:
        config.callbacks.append(
            PrintProgress(
                max_time=progress_max_time,
                start_time=progress_start_time,
                print_env_info=progress_print_env_info,
                print_worker_info=progress_print_worker_info,
                print_train_info=progress_print_train_info,
                print_worker=progress_print_worker,
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
    _, parameter, memory, _ = play(config, parameter, remote_memory)

    # history
    history = FileLogPlot()
    try:
        if file_logger is not None:
            history.load(file_logger.base_dir)
    except Exception:
        logger.warning(traceback.format_exc())

    return parameter, memory, history


def evaluate(
    config: Config,
    parameter: Optional[RLParameter] = None,
    # stop reason
    max_episodes: int = 10,
    timeout: int = -1,
    max_steps: int = -1,
    # play config
    shuffle_player: bool = False,
    seed: Optional[int] = None,
    # PrintProgress
    print_progress: bool = False,
    progress_max_time: int = 60 * 5,  # s
    progress_start_time: int = 5,
    progress_print_env_info: bool = False,
    progress_print_worker_info: bool = True,
    progress_print_worker: int = 0,
    # other
    callbacks: List[Callback] = [],
    remote_memory: Optional[RLRemoteMemory] = None,
) -> Union[List[float], List[List[float]]]:  # single play , multi play
    callbacks = callbacks[:]

    assert (
        max_steps != -1 or max_episodes != -1 or timeout != -1
    ), "Please specify 'max_episodes', 'timeout' or 'max_steps'."

    config = config.copy(env_share=True)
    # stop config
    config.max_steps = max_steps
    config.max_episodes = max_episodes
    config.timeout = timeout
    # play config
    config.shuffle_player = shuffle_player
    config.disable_trainer = True
    if config.seed is None:
        config.seed = seed
    # evaluate
    config.enable_evaluation = False
    # callbacks
    config.callbacks = callbacks
    # play info
    config.training = False
    config.distributed = False

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
            )
        )

    # play
    episode_rewards, parameter, memory, env = play(config, parameter, remote_memory)

    if env.player_num == 1:
        return [r[0] for r in episode_rewards]
    else:
        return episode_rewards


def render(
    config: Config,
    parameter: Optional[RLParameter] = None,
    # Rendering
    render_terminal: bool = True,
    render_window: bool = False,
    render_kwargs: dict = {},
    step_stop: bool = False,
    enable_animation: bool = False,
    use_skip_step: bool = True,
    # stop config
    max_steps: int = -1,
    timeout: int = -1,
    seed: Optional[int] = None,
    # PrintProgress
    print_progress: bool = False,
    progress_max_time: int = 60 * 5,  # s
    progress_start_time: int = 5,
    progress_print_env_info: bool = False,
    progress_print_worker_info: bool = True,
    progress_print_worker: int = 0,
    # other
    callbacks: List[Callback] = [],
    remote_memory: Optional[RLRemoteMemory] = None,
) -> Tuple[List[float], Rendering]:
    callbacks = callbacks[:]
    _render_kwargs = {}
    _render_kwargs.update(render_kwargs)
    render_kwargs = _render_kwargs

    config = config.copy(env_share=True)
    # stop config
    config.max_episodes = 1
    config.timeout = timeout
    config.max_steps = max_steps
    config.max_train_count = -1
    # play config
    config.shuffle_player = False
    config.disable_trainer = True
    if config.seed is None:
        config.seed = seed
    # evaluate
    config.enable_evaluation = False
    # callbacks
    config.callbacks = callbacks
    # play info
    config.training = False
    config.distributed = False

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
            )
        )

    # Rendering
    render = Rendering(
        render_terminal=render_terminal,
        render_window=render_window,
        render_kwargs=render_kwargs,
        step_stop=step_stop,
        enable_animation=enable_animation,
        use_skip_step=use_skip_step,
    )
    config.callbacks.append(render)

    # play
    episode_rewards, parameter, memory, env = play(config, parameter, remote_memory)

    return episode_rewards[0], render


def animation(
    config: Config,
    parameter: Optional[RLParameter] = None,
    # Rendering
    render_kwargs: dict = {},
    use_skip_step: bool = True,
    # stop config
    max_steps: int = -1,
    timeout: int = -1,
    seed: Optional[int] = None,
    # PrintProgress
    print_progress: bool = False,
    progress_max_time: int = 60 * 5,  # s
    progress_start_time: int = 5,
    progress_print_env_info: bool = False,
    progress_print_worker_info: bool = True,
    progress_print_worker: int = 0,
    # other
    callbacks: List[Callback] = [],
    remote_memory: Optional[RLRemoteMemory] = None,
) -> Rendering:
    rewards, anime = render(
        config=config,
        parameter=parameter,
        render_terminal=False,
        render_window=False,
        render_kwargs=render_kwargs,
        step_stop=False,
        enable_animation=True,
        use_skip_step=use_skip_step,
        max_steps=max_steps,
        timeout=timeout,
        seed=seed,
        print_progress=print_progress,
        progress_max_time=progress_max_time,
        progress_start_time=progress_start_time,
        progress_print_env_info=progress_print_env_info,
        progress_print_worker_info=progress_print_worker_info,
        progress_print_worker=progress_print_worker,
        callbacks=callbacks,
        remote_memory=remote_memory,
    )
    return anime
