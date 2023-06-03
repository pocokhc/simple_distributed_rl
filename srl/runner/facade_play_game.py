import logging
from typing import Any, List, Optional, Union

from srl.base.env.config import EnvConfig
from srl.base.rl.base import RLConfig, RLRemoteMemory
from srl.runner.callback import GameCallback

logger = logging.getLogger(__name__)


def play(
    env_config: Union[str, EnvConfig],
    key_bind: Any = None,
    action_division_num: int = 5,
    # worker memory
    rl_config: Optional[RLConfig] = None,
    # other
    callbacks: List[GameCallback] = [],
) -> Optional[RLRemoteMemory]:
    from srl.utils.common import is_packages_installed

    error_text = "This run requires installation of 'PIL', 'pygame'. "
    error_text += "(pip install pillow pygame)"
    assert is_packages_installed(["PIL", "pygame"]), error_text

    from srl.runner.game_window import PlayableGame

    game = PlayableGame(
        env_config,
        key_bind,
        action_division_num,
        rl_config,
        callbacks=callbacks,
    )
    game.play()

    return game.remote_memory
