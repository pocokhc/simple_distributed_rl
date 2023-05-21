import logging
from typing import List, Optional, Tuple, Union, cast

from srl.base.define import PlayRenderModes
from srl.base.rl.base import RLParameter, RLRemoteMemory
from srl.runner.callback import Callback
from srl.runner.callbacks.history_viewer import HistoryViewer
from srl.runner.config import Config
from srl.runner.core import CheckpointOption, EvalOption, HistoryOption, ProgressOption, play

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
    # option
    eval: Optional[EvalOption] = None,
    progress: Optional[ProgressOption] = ProgressOption(),
    history: Optional[HistoryOption] = HistoryOption(
        write_memory=True,
        write_file=False,
    ),
    checkpoint: Optional[CheckpointOption] = None,
    # other
    callbacks: List[Callback] = [],
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
) -> Tuple[RLParameter, RLRemoteMemory, HistoryViewer]:
    _, parameter, remote_memory, return_history = play(
        config,
        # stop config
        max_episodes=max_episodes,
        timeout=timeout,
        max_steps=max_steps,
        max_train_count=max_train_count,
        # play config
        train_only=False,
        shuffle_player=shuffle_player,
        disable_trainer=disable_trainer,
        enable_profiling=enable_profiling,
        # play info
        training=True,
        distributed=False,
        render_mode=PlayRenderModes.none,
        # option
        eval=eval,
        progress=progress,
        history=history,
        checkpoint=checkpoint,
        # other
        callbacks=callbacks,
        parameter=parameter,
        remote_memory=remote_memory,
    )
    return parameter, remote_memory, return_history


def train_only(
    config: Config,
    remote_memory: Optional[RLRemoteMemory] = None,
    # stop config
    max_train_count: int = -1,
    timeout: int = -1,
    # play config
    enable_profiling: bool = True,
    # option
    eval: Optional[EvalOption] = None,
    progress: Optional[ProgressOption] = ProgressOption(),
    history: Optional[HistoryOption] = None,
    checkpoint: Optional[CheckpointOption] = None,
    # other
    callbacks: List[Callback] = [],
    parameter: Optional[RLParameter] = None,
) -> Tuple[RLParameter, RLRemoteMemory, HistoryViewer]:
    _, parameter, remote_memory, return_history = play(
        config,
        # stop config
        max_episodes=-1,
        timeout=timeout,
        max_steps=-1,
        max_train_count=max_train_count,
        # play config
        train_only=True,
        shuffle_player=False,
        disable_trainer=False,
        enable_profiling=enable_profiling,
        # play info
        training=True,
        distributed=False,
        render_mode=PlayRenderModes.none,
        # option
        eval=eval,
        progress=progress,
        history=history,
        checkpoint=checkpoint,
        # other
        callbacks=callbacks,
        parameter=parameter,
        remote_memory=remote_memory,
    )
    return parameter, remote_memory, return_history


def evaluate(
    config: Config,
    parameter: Optional[RLParameter] = None,
    # stop reason
    max_episodes: int = 10,
    timeout: int = -1,
    max_steps: int = -1,
    # play config
    shuffle_player: bool = False,
    # option
    progress: Optional[ProgressOption] = ProgressOption(),
    # other
    callbacks: List[Callback] = [],
    remote_memory: Optional[RLRemoteMemory] = None,
) -> Union[List[float], List[List[float]]]:  # single play , multi play
    config._run_name = "eval"
    episode_rewards, _, _, _ = play(
        config,
        # stop config
        max_episodes=max_episodes,
        timeout=timeout,
        max_steps=max_steps,
        max_train_count=-1,
        # play config
        train_only=False,
        shuffle_player=shuffle_player,
        disable_trainer=True,
        enable_profiling=False,
        # play info
        training=False,
        distributed=False,
        render_mode=PlayRenderModes.none,
        # option
        eval=None,
        progress=progress,
        history=None,
        checkpoint=None,
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
    mode: PlayRenderModes = PlayRenderModes.terminal,
    render_kwargs: dict = {},
    step_stop: bool = False,
    render_skip_step: bool = True,
    # play config
    timeout: int = -1,
    max_steps: int = -1,
    # option
    progress: Optional[ProgressOption] = None,
    # other
    callbacks: List[Callback] = [],
    remote_memory: Optional[RLRemoteMemory] = None,
) -> List[float]:
    callbacks = callbacks[:]

    from srl.runner.callbacks.rendering import Rendering

    callbacks.append(
        Rendering(
            mode=mode,
            kwargs=render_kwargs,
            step_stop=step_stop,
            render_skip_step=render_skip_step,
        )
    )

    episode_rewards, _, _, _ = play(
        config,
        # stop config
        max_episodes=1,
        timeout=timeout,
        max_steps=max_steps,
        max_train_count=-1,
        # play config
        train_only=False,
        shuffle_player=False,
        disable_trainer=True,
        enable_profiling=False,
        # play info
        training=False,
        distributed=False,
        render_mode=mode,
        # option
        eval=None,
        progress=progress,
        history=None,
        checkpoint=None,
        # other
        callbacks=callbacks,
        parameter=parameter,
        remote_memory=remote_memory,
    )
    return episode_rewards[0]


def render_window(
    config: Config,
    parameter: Optional[RLParameter] = None,
    # Rendering
    mode: PlayRenderModes = PlayRenderModes.window,
    render_kwargs: dict = {},
    step_stop: bool = False,
    render_skip_step: bool = True,
    # play config
    timeout: int = -1,
    max_steps: int = -1,
    # option
    progress: Optional[ProgressOption] = None,
    # other
    callbacks: List[Callback] = [],
    remote_memory: Optional[RLRemoteMemory] = None,
) -> List[float]:
    callbacks = callbacks[:]

    from srl.runner.callbacks.rendering import Rendering

    callbacks.append(
        Rendering(
            mode=mode,
            kwargs=render_kwargs,
            step_stop=step_stop,
            render_skip_step=render_skip_step,
        )
    )

    episode_rewards, _, _, _ = play(
        config,
        # stop config
        max_episodes=1,
        timeout=timeout,
        max_steps=max_steps,
        max_train_count=-1,
        # play config
        train_only=False,
        shuffle_player=False,
        disable_trainer=True,
        enable_profiling=False,
        # play info
        training=False,
        distributed=False,
        render_mode=mode,
        # option
        eval=None,
        progress=progress,
        history=None,
        checkpoint=None,
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
    step_stop: bool = False,
    render_skip_step: bool = True,
    # play config
    max_steps: int = -1,
    timeout: int = -1,
    # option
    progress: Optional[ProgressOption] = ProgressOption(),
    # other
    callbacks: List[Callback] = [],
):
    callbacks = callbacks[:]

    from srl.runner.callbacks.rendering import Rendering

    mode = PlayRenderModes.rgb_array
    rendering = Rendering(
        mode=mode,
        kwargs=render_kwargs,
        step_stop=step_stop,
        render_skip_step=render_skip_step,
    )
    callbacks.append(rendering)

    _, _, _, _ = play(
        config,
        # stop config
        max_episodes=1,
        timeout=timeout,
        max_steps=max_steps,
        max_train_count=-1,
        # play config
        train_only=False,
        shuffle_player=False,
        disable_trainer=True,
        enable_profiling=False,
        # play info
        training=False,
        distributed=False,
        render_mode=mode,
        # option
        eval=None,
        progress=progress,
        history=None,
        checkpoint=None,
        # other
        callbacks=callbacks,
        parameter=parameter,
    )
    return cast(Rendering, rendering)


def replay(
    config: Config,
    parameter: Optional[RLParameter] = None,
    # play config
    max_episodes: int = 5,
    timeout: int = -1,
    max_steps: int = -1,
    # option
    progress: Optional[ProgressOption] = ProgressOption(),
    # other
    callbacks: List[Callback] = [],
):
    callbacks = callbacks[:]

    from srl.runner.callbacks.history_episode import HistoryEpisode

    history = HistoryEpisode(save_dir=config.save_dir)
    callbacks.append(history)

    play(
        config,
        # stop config
        max_episodes=max_episodes,
        timeout=timeout,
        max_steps=max_steps,
        max_train_count=-1,
        # play config
        train_only=False,
        shuffle_player=False,
        disable_trainer=True,
        enable_profiling=False,
        # play info
        training=False,
        distributed=False,
        render_mode=PlayRenderModes.rgb_array,
        # option
        eval=None,
        progress=progress,
        history=None,
        checkpoint=None,
        # other
        callbacks=callbacks,
        parameter=parameter,
    )
    history.replay()
    return history
