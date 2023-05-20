from typing import List, Optional, Tuple

from srl.base.rl.base import RLParameter, RLRemoteMemory
from srl.runner.callback import Callback
from srl.runner.callbacks.history_viewer import HistoryViewer
from srl.runner.config import Config
from srl.runner.core import CheckpointOption, EvalOption, HistoryOption, ProgressOption


def train_remote(
    config: Config,
    # stop config
    max_episodes: int = -1,
    timeout: int = -1,
    max_steps: int = -1,
    max_train_count: int = -1,
    # play config
    shuffle_player: bool = True,
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
    queue_timeout: int = 60 * 10,
) -> Tuple[RLParameter, RLRemoteMemory, HistoryViewer]:
    from srl.runner.core_remote import train

    return train(
        config,
        max_episodes=max_episodes,
        timeout=timeout,
        max_steps=max_steps,
        max_train_count=max_train_count,
        shuffle_player=shuffle_player,
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
        queue_timeout=queue_timeout,
    )


def run_actor(
    server_ip: str,
    port: int,
    authkey: bytes = b"abracadabra",
    actor_id: Optional[int] = None,
    verbose: bool = True,
):
    from srl.runner.core_remote import run_actor

    run_actor(server_ip, port, authkey, actor_id, verbose)
