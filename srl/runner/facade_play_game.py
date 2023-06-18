import logging
from typing import Any, List, Optional, Union

from srl.base.env.config import EnvConfig
from srl.base.rl.base import RLConfig, RLRemoteMemory
from srl.runner.callback import GameCallback
from srl.runner.config import Config as RunnerConfig

logger = logging.getLogger(__name__)


def play_window(
    config: Union[str, EnvConfig, RunnerConfig],
    key_bind: Any = None,
    action_division_num: int = 5,
    # worker memory
    rl_config: Optional[RLConfig] = None,
    # other
    callbacks: List[GameCallback] = [],
    _is_test: bool = False,  # for test
) -> Optional[RLRemoteMemory]:
    from srl.utils.common import is_packages_installed

    error_text = "This run requires installation of 'PIL', 'pygame'. "
    error_text += "(pip install pillow pygame)"
    assert is_packages_installed(["PIL", "pygame"]), error_text

    if isinstance(config, RunnerConfig):
        config = config.env_config

    from srl.runner.game_windows.playable_game import PlayableGame

    game = PlayableGame(
        config,
        key_bind,
        action_division_num,
        rl_config,
        callbacks=callbacks,
        _is_test=_is_test,
    )
    game.play()

    return game.remote_memory
