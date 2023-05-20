from typing import List, Optional, Tuple

from srl.base.rl.base import RLParameter, RLRemoteMemory
from srl.runner.callback import Callback
from srl.runner.callbacks.history_viewer import HistoryViewer
from srl.runner.config import Config
from srl.runner.core import CheckpointOption, EvalOption, HistoryOption, ProgressOption


def train_mp(
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
    # options
    eval: Optional[EvalOption] = None,
    progress: Optional[ProgressOption] = ProgressOption(),
    history: Optional[HistoryOption] = None,
    checkpoint: Optional[CheckpointOption] = None,
    # other
    callbacks: List[Callback] = [],
    parameter: Optional[RLParameter] = None,
    remote_memory: Optional[RLRemoteMemory] = None,
    return_remote_memory: bool = False,
    save_remote_memory: str = "",
) -> Tuple[RLParameter, RLRemoteMemory, HistoryViewer]:
    from srl.runner.core_mp import train

    return train(
        config,
        max_episodes=max_episodes,
        timeout=timeout,
        max_steps=max_steps,
        max_train_count=max_train_count,
        shuffle_player=shuffle_player,
        disable_trainer=disable_trainer,
        enable_profiling=enable_profiling,
        eval=eval,
        progress=progress,
        history=history,
        checkpoint=checkpoint,
        callbacks=callbacks,
        parameter=parameter,
        remote_memory=remote_memory,
        return_remote_memory=return_remote_memory,
        save_remote_memory=save_remote_memory,
    )
