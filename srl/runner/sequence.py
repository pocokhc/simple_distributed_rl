import logging
from typing import List, Optional, Tuple, Union

from srl.base.define import PlayRenderMode
from srl.base.rl.base import RLConfig, RLParameter, RLRemoteMemory
from srl.runner.callback import Callback
from srl.runner.callbacks.file_log_reader import FileLogReader
from srl.runner.config import Config
from srl.runner.play_sequence import play_facade

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
    # file_log
    enable_file_logger: bool = False,
    file_logger_tmp_dir: str = "tmp",
    file_logger_enable_train_log: bool = True,
    file_logger_train_log_interval: int = 1,  # s
    file_logger_enable_episode_log: bool = False,
    file_logger_enable_checkpoint: bool = True,
    file_logger_checkpoint_interval: int = 60 * 20,  # s
    # other
    callbacks: List[Callback] = [],
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
) -> Tuple[RLParameter, RLRemoteMemory, FileLogReader]:
    _, parameter, remote_memory, history, _ = play_facade(
        config,
        # stop config
        max_episodes=max_episodes,
        timeout=timeout,
        max_steps=max_steps,
        max_train_count=max_train_count,
        # play config
        shuffle_player=shuffle_player,
        disable_trainer=disable_trainer,
        enable_profiling=enable_profiling,
        # evaluate
        enable_evaluation=enable_evaluation,
        eval_env_sharing=eval_env_sharing,
        eval_interval=eval_interval,
        eval_num_episode=eval_num_episode,
        eval_players=eval_players,
        # play info
        training=True,
        distributed=False,
        render_mode=PlayRenderMode.none,
        render_kwargs={},
        # PrintProgress
        print_progress=print_progress,
        progress_max_time=progress_max_time,
        progress_start_time=progress_start_time,
        progress_print_env_info=progress_print_env_info,
        progress_print_worker_info=progress_print_worker_info,
        progress_print_train_info=progress_print_train_info,
        progress_print_worker=progress_print_worker,
        # file_log
        enable_file_logger=enable_file_logger,
        file_logger_tmp_dir=file_logger_tmp_dir,
        file_logger_enable_train_log=file_logger_enable_train_log,
        file_logger_train_log_interval=file_logger_train_log_interval,
        file_logger_enable_episode_log=file_logger_enable_episode_log,
        file_logger_episode_log_add_render=False,
        file_logger_enable_checkpoint=file_logger_enable_checkpoint,
        file_logger_checkpoint_interval=file_logger_checkpoint_interval,
        # Rendering
        enable_rendering=False,
        # other
        callbacks=callbacks,
        parameter=parameter,
        remote_memory=remote_memory,
    )
    return parameter, remote_memory, history


def evaluate(
    config: Config,
    parameter: Optional[RLParameter] = None,
    # stop reason
    max_episodes: int = 10,
    timeout: int = -1,
    max_steps: int = -1,
    # play config
    shuffle_player: bool = False,
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
    episode_rewards, _, _, _, _ = play_facade(
        config,
        # stop config
        max_episodes=max_episodes,
        timeout=timeout,
        max_steps=max_steps,
        max_train_count=-1,
        # play config
        shuffle_player=shuffle_player,
        disable_trainer=True,
        enable_profiling=False,
        # evaluate
        enable_evaluation=False,
        # play info
        training=False,
        distributed=False,
        # render mode
        render_mode=PlayRenderMode.none,
        render_kwargs={},
        # PrintProgress
        print_progress=print_progress,
        progress_max_time=progress_max_time,
        progress_start_time=progress_start_time,
        progress_print_env_info=progress_print_env_info,
        progress_print_worker_info=progress_print_worker_info,
        progress_print_train_info=False,
        progress_print_worker=progress_print_worker,
        # file_log
        enable_file_logger=False,
        # Rendering
        enable_rendering=False,
        # other
        callbacks=callbacks,
        parameter=parameter,
        remote_memory=remote_memory,
    )
    if config.env_config.player_num == 1:
        return [r[0] for r in episode_rewards]
    else:
        return episode_rewards


def render(
    config: Config,
    parameter: Optional[RLParameter] = None,
    # Rendering
    render_mode: Union[str, PlayRenderMode] = PlayRenderMode.terminal,
    render_kwargs: dict = {},
    step_stop: bool = False,
    use_skip_step: bool = True,
    # play config
    timeout: int = -1,
    max_steps: int = -1,
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
) -> List[float]:
    episode_rewards, _, _, _, _ = play_facade(
        config,
        # stop config
        max_episodes=1,
        timeout=timeout,
        max_steps=max_steps,
        max_train_count=-1,
        # play config
        shuffle_player=False,
        disable_trainer=True,
        # render mode
        render_mode=render_mode,
        render_kwargs=render_kwargs,
        # evaluate
        enable_evaluation=False,
        # play info
        training=False,
        distributed=False,
        # PrintProgress
        print_progress=print_progress,
        progress_max_time=progress_max_time,
        progress_start_time=progress_start_time,
        progress_print_env_info=progress_print_env_info,
        progress_print_worker_info=progress_print_worker_info,
        progress_print_train_info=False,
        progress_print_worker=progress_print_worker,
        # file_log
        enable_file_logger=False,
        # Rendering
        enable_rendering=True,
        step_stop=step_stop,
        use_skip_step=use_skip_step,
        # other
        callbacks=callbacks,
        parameter=parameter,
        remote_memory=remote_memory,
    )
    return episode_rewards[0]


def animation(
    config: Config,
    parameter: Optional[RLParameter] = None,
    # Rendering
    render_kwargs: dict = {},
    use_skip_step: bool = True,
    # play config
    timeout: int = -1,
    max_steps: int = -1,
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
) -> "srl.runner.callbacks.rendering.Rendering":
    episode_rewards, _, _, _, _render = play_facade(
        config,
        # stop config
        max_episodes=1,
        timeout=timeout,
        max_steps=max_steps,
        max_train_count=-1,
        # play config
        shuffle_player=False,
        disable_trainer=True,
        # render mode
        render_mode=PlayRenderMode.rgb_array,
        render_kwargs=render_kwargs,
        # evaluate
        enable_evaluation=False,
        # play info
        training=False,
        distributed=False,
        # PrintProgress
        print_progress=print_progress,
        progress_max_time=progress_max_time,
        progress_start_time=progress_start_time,
        progress_print_env_info=progress_print_env_info,
        progress_print_worker_info=progress_print_worker_info,
        progress_print_train_info=False,
        progress_print_worker=progress_print_worker,
        # file_log
        enable_file_logger=False,
        # Rendering
        enable_rendering=True,
        step_stop=False,
        use_skip_step=use_skip_step,
        # other
        callbacks=callbacks,
        parameter=parameter,
        remote_memory=remote_memory,
    )
    logger.info(f"animation rewards: {episode_rewards[0]}")
    return _render


def test_play(
    config: Config,
    parameter: Optional[RLParameter] = None,
    # stop config
    max_episodes: int = 5,
    timeout: int = -1,
    max_steps: int = -1,
    # play config
    shuffle_player: bool = False,
    enable_profiling: bool = False,
    # render mode
    render_mode: Union[str, PlayRenderMode] = PlayRenderMode.rgb_array,
    render_kwargs: dict = {},
    # PrintProgress
    print_progress: bool = False,
    progress_max_time: int = 60 * 5,  # s
    progress_start_time: int = 5,
    progress_print_env_info: bool = False,
    progress_print_worker_info: bool = True,
    progress_print_train_info: bool = False,
    progress_print_worker: int = 0,
    # file_log
    file_logger_tmp_dir: str = "tmp",
    # other
    callbacks: List[Callback] = [],
    remote_memory: Optional[RLRemoteMemory] = None,
) -> FileLogReader:
    _, parameter, memory, history, _ = play_facade(
        config,
        # stop config
        max_episodes=max_episodes,
        timeout=timeout,
        max_steps=max_steps,
        max_train_count=-1,
        # play config
        shuffle_player=shuffle_player,
        disable_trainer=True,
        enable_profiling=enable_profiling,
        # evaluate
        enable_evaluation=False,
        # play info
        training=False,
        distributed=False,
        # render mode
        render_mode=render_mode,
        render_kwargs=render_kwargs,
        # PrintProgress
        print_progress=print_progress,
        progress_max_time=progress_max_time,
        progress_start_time=progress_start_time,
        progress_print_env_info=progress_print_env_info,
        progress_print_worker_info=progress_print_worker_info,
        progress_print_train_info=progress_print_train_info,
        progress_print_worker=progress_print_worker,
        # file_log
        enable_file_logger=True,
        file_logger_tmp_dir=file_logger_tmp_dir,
        file_logger_enable_train_log=False,
        file_logger_enable_episode_log=True,
        file_logger_episode_log_add_render=True,
        file_logger_enable_checkpoint=False,
        # Rendering
        enable_rendering=False,
        step_stop=False,
        use_skip_step=True,
        # other
        callbacks=callbacks,
        parameter=parameter,
        remote_memory=remote_memory,
    )
    return history
